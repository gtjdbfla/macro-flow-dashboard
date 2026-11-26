"""자료 신선도 점검 — 자동 수집 자료가 발표 주기에 비해 오래됐는지, 손으로 관리하는 항목이 낡았는지 한눈에 본다. 읽기 전용.
컨테이너에서 실행: docker compose exec -T macro-flow-dashboard python scripts/audit_freshness.py
각 줄은 '상태 | 항목 | 최신 날짜(경과일) | 원인과 할 일' 형식이다. 기준 일수는 발표 주기에 맞춘 운영 규칙이다."""
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, PARQUET  # noqa: E402

TODAY = pd.Timestamp.now().normalize()
WEEKLY_FRED = {"WALCL", "WTREGEN", "NFCI", "IC4WSA"}
MONTHLY_FRED = {"SAHMREALTIME"}
QUARTERLY_V = {"GDPC1", "CP"}
# 수집 목록(src/data/universe.py의 FRED_DAILY·FRED_VINTAGE, scripts/collect.py)에 없는 자료는 한 번 받은 뒤 갱신되지 않는다.
NOT_COLLECTED = {"px__KS200": "과거 수집 잔재(이력이 없는 티커라 KODEX200으로 대체). 사용하지 않음"}


def max_age(name):
    """파일 이름으로 발표 주기에 맞는 허용 일수. 점검하지 않는 자료는 None."""
    if name.startswith("fredv_"):
        return 150 if name[6:] in QUARTERLY_V else 80
    if name.startswith("fred_"):
        sid = name[5:]
        return 80 if sid in MONTHLY_FRED else 14 if sid in WEEKLY_FRED else 6
    if name.startswith(("px_", "ssga_", "kis_", "cboe_", "token_prices")):
        return 6
    if name.startswith("cftc"):
        return 14
    if name.startswith("french"):
        return 75
    if name.startswith("twse"):
        return 60
    if name == "cloud_segments":
        return 130
    return None


def last_date(path):
    d = pd.read_parquet(path)
    for c in ("date", "period_end", "quarter_end", "fetched", "month", "as_of"):
        if c in d.columns:
            v = pd.to_datetime(d[c], errors="coerce").max()
            if pd.notna(v):
                return v
    if isinstance(d.index, pd.DatetimeIndex) and len(d):
        return d.index.max()
    return None


def audit_parquet():
    out = []
    for f in sorted(PARQUET.glob("*.parquet")):
        name = f.stem
        limit = max_age(name)
        if limit is None:
            continue
        try:
            last = last_date(f)
        except Exception as e:
            out.append(("오류", name, "-", f"읽기 실패 {type(e).__name__}"))
            continue
        if last is None:
            continue
        age = (TODAY - last).days
        if age > limit:
            why = NOT_COLLECTED.get(name, f"발표 주기 기준 {limit}일 초과. 수집 오류 또는 발표 지연 확인")
            out.append(("지연", name, f"{last.date()}({age}일)", why))
    return out


def audit_sources():
    out = []
    try:
        cov = json.loads((DATA_DIR / "coverage.json").read_text(encoding="utf-8"))
    except Exception:
        return [("오류", "coverage.json", "-", "읽을 수 없음")]
    for k, v in cov.items():
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(v["updated"])).total_seconds() / 3600 if v.get("updated") else None
        series = v.get("series", {})
        err = [n for n, x in series.items() if isinstance(x, dict) and "error" in x]
        if v.get("error") or err:
            out.append(("오류", f"수집:{k}", v.get("updated", "")[:16], v.get("error", f"시리즈 오류 {len(err)}개: {err[:3]}")[:120]))
        # 주 1회·월 1회 수집 소스는 간격이 길다
        slow = {"french": 24 * 9, "twse": 24 * 35}.get(k, 24 * 4)
        if age is not None and age > slow:
            out.append(("지연", f"수집:{k}", v["updated"][:16], f"마지막 수집이 {age / 24:.1f}일 전. 크론 확인"))
    return out


def manual_dates():
    """손으로 적은 표의 가장 최근 날짜와 지난 일정."""
    out = []
    text = (ROOT / "src/ai/end_demand.py").read_text(encoding="utf-8")
    m = re.search(r"CRWV_STRESS\s*=\s*\((.*?)\n\)", text, re.S)
    if m:
        ds = sorted(re.findall(r'\("(\d{4}-\d{2}-\d{2})"', m.group(1)))
        if ds and (TODAY - pd.Timestamp(ds[-1])).days > 7:
            out.append(("수기", "CoreWeave 신용 스트레스(CRWV_STRESS)", f"{ds[-1]}({(TODAY - pd.Timestamp(ds[-1])).days}일)",
                        "새 보도를 검색해 원문 숫자를 직접 확인한 경우에만 행 추가"))
    core = (ROOT / "views/core.py").read_text(encoding="utf-8")
    m = re.search(r"EVENTS\s*=\s*\((.*?)\n\)", core, re.S)
    if m:
        for d in re.findall(r'\("(\d{4}-\d{2}(?:-\d{2})?)', m.group(1)):
            day = pd.Timestamp(d + "-28") if len(d) == 7 else pd.Timestamp(d)
            if day < TODAY:
                out.append(("수기", "핵심 현황 일정(views/core.py EVENTS)", d, "지난 일정. 결과를 확인하고 목록에서 빼거나 다음 일정으로 교체"))
    return out


def blind_spots():
    """화면에서 '판정 불가·미입력·보류'로 보이는 항목과 이유."""
    out = []
    try:
        g = json.loads((DATA_DIR / "results/ai_guide.json").read_text(encoding="utf-8"))
        for i in g["snapshot"]["bubble"]["resolution"] + g["snapshot"]["bubble"]["warning"]:
            if i["ok"] is None:
                out.append(("판정불가", f"점검표 {i['id']}", "-", i["detail"][:110]))
        for f in json.loads((DATA_DIR / "results/ai_flags.json").read_text(encoding="utf-8"))["flags"]:
            if f["level"] == "보류":
                out.append(("보류", f"신호 {f['id']}", "-", f["text"][:110]))
    except Exception as e:
        out.append(("오류", "ai_guide.json", "-", type(e).__name__))
    try:
        ew = json.loads((DATA_DIR / "results/ai_early_warning.json").read_text(encoding="utf-8"))
        for i in ew["indicators"]:
            if i["level"] == "na":
                out.append(("미입력", f"조기경보 {i['id']} {i['name']}", "-", i["detail"][:110]))
    except Exception:
        pass
    return out


def run():
    rows = audit_sources() + audit_parquet() + manual_dates() + blind_spots()
    print(f"자료 신선도 점검 {TODAY.date()}: 항목 {len(rows)}개")
    for r in rows:
        print(" | ".join(r))
    if not rows:
        print("문제 없음")
    return rows


if __name__ == "__main__":
    run()
