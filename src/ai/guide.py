"""투자 길잡이용 상태 스냅샷과 주간 변화. 신호·수치는 기존 계산 결과를 그대로 모으기만 하고 새 예측이나 종합 점수는 만들지 않는다.
개선 점검표(CHECKLIST)는 규칙을 docs/AI_PROFITABILITY.md(길잡이 절)에 먼저 적어 둔 예/아니오 항목이며 항목 수 합계에 예측력이 있다고 주장하지 않는다."""
import json

import pandas as pd

SNAPSHOT_MIN_DAYS = 6
LEVEL_RANK = {"긍정": 2, "강함": 2, "정상": 2, "일치": 2, "중립": 1, "보류": 0, "주의": -1, "둔화": -1, "경고": -2}

# (id, 설명, 분기 지표 이름). 최근 두 분기말의 Big4 합계를 비교한다. 전부 후행 지표라 분기 실적이 나올 때만 바뀐다.
CHECKLIST = (
    ("capex_decel", "Big4 CAPEX 증가율이 직전 분기보다 낮아졌다", "capex_yoy", "down"),
    ("capex_ocf_down", "Big4 CAPEX ÷ 영업현금흐름이 직전 분기보다 낮아졌다", "capex_ocf", "down"),
    ("fcf_up", "Big4 12개월 잉여현금흐름이 직전 분기보다 늘었다", "fcf_ttm", "up"),
    ("payback_up", "영업이익 증가액 ÷ 1년 전 CAPEX가 직전 분기보다 높아졌다", "opinc_per_capex", "up"),
)
FLAG_CHECKS = (
    ("cloud_ok", "클라우드 성장·수익성(F7)이 긍정이다", "F7", {"긍정"}),
    ("software_ok", "소프트웨어 AI 매출화(F18)가 긍정이다", "F18", {"긍정"}),
    ("nvda_cash_ok", "NVIDIA 수요의 질(F16)이 정상이다", "F16", {"정상"}),
    ("borrowers_ok", "차입 의존 기업 런웨이(F11)가 경고가 아니다", "F11", {"긍정", "정상", "중립", "주의", "보류"}),
)


def checklist(big4, flags):
    """big4: ai_history의 BIG4 행. flags: {id: level}."""
    h = big4.sort_values("period_end").dropna(subset=["capex_yoy"]).tail(2)
    items = []
    for key, text, col, direction in CHECKLIST:
        if len(h) < 2 or h[col].isna().any():
            items.append({"id": key, "text": text, "ok": None, "detail": "자료 부족"})
            continue
        prev, cur = float(h[col].iloc[0]), float(h[col].iloc[1])
        ok = cur < prev if direction == "down" else cur > prev
        items.append({"id": key, "text": text, "ok": bool(ok),
                      "detail": f"{h.period_end.iloc[0]:%Y-%m} {prev:,.3g} → {h.period_end.iloc[1]:%Y-%m} {cur:,.3g}"})
    for key, text, fid, good in FLAG_CHECKS:
        lvl = flags.get(fid)
        items.append({"id": key, "text": text, "ok": None if lvl is None else lvl in good, "detail": f"{fid} {lvl or '없음'}"})
    return items


def snapshot(date, risk, flags, big4, metrics=None):
    """한 날짜의 상태. risk=risk_gauge.json, flags=ai_flags.json의 flags 목록."""
    levels = {f["id"]: f["level"] for f in flags}
    macro = risk.get("macro", {})
    return {
        "date": str(date),
        "macro": {"outlook": macro.get("outlook"), "as_of": macro.get("as_of"),
                  "metrics": {k: {"state": v["state"], "value": v["value"], "change": v["change"]}
                              for k, v in macro.get("metrics", {}).items()}},
        "market": {k: risk.get(k) for k in ("as_of", "score", "level", "spx", "spx_dd", "vix", "shock", "kospi_level", "kospi_dd")},
        "flags": levels,
        "checklist": checklist(big4, levels),
        "metrics": metrics or {},
    }


def append(path, snap):
    """같은 날짜의 스냅샷은 덮어쓴다. 장부는 날짜 오름차순 JSON Lines."""
    rows = []
    if path.exists():
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["date"] != snap["date"]] + [snap]
    rows.sort(key=lambda r: r["date"])
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return rows


def compare(rows, min_days=SNAPSHOT_MIN_DAYS):
    """가장 최근 스냅샷과 min_days 이상 전의 가장 가까운 스냅샷을 비교한다. 없으면 이전 스냅샷 중 가장 오래된 것."""
    cur = rows[-1]
    older = [r for r in rows[:-1] if (pd.Timestamp(cur["date"]) - pd.Timestamp(r["date"])).days >= min_days]
    prev = older[-1] if older else (rows[0] if len(rows) > 1 else None)
    if prev is None:
        return {"base": None, "flag_changes": [], "macro_changes": [], "checklist_changes": [], "market": {}}
    flag_changes = []
    for fid, lvl in cur["flags"].items():
        before = prev["flags"].get(fid)
        if before is not None and before != lvl:
            flag_changes.append({"id": fid, "from": before, "to": lvl,
                                 "direction": "개선" if LEVEL_RANK.get(lvl, 0) > LEVEL_RANK.get(before, 0) else "악화"
                                 if LEVEL_RANK.get(lvl, 0) < LEVEL_RANK.get(before, 0) else "변화"})
    macro_changes = []
    if cur["macro"]["outlook"] != prev["macro"]["outlook"]:
        macro_changes.append({"name": "종합 판정", "from": prev["macro"]["outlook"], "to": cur["macro"]["outlook"]})
    for k, v in cur["macro"]["metrics"].items():
        before = prev["macro"]["metrics"].get(k)
        if before and before["state"] != v["state"]:
            macro_changes.append({"name": k, "from": before["state"], "to": v["state"]})
    checklist_changes = []
    before_items = {i["id"]: i["ok"] for i in prev["checklist"]}
    for item in cur["checklist"]:
        if item["id"] in before_items and before_items[item["id"]] != item["ok"]:
            checklist_changes.append({"id": item["id"], "text": item["text"], "from": before_items[item["id"]], "to": item["ok"]})
    market = {}
    for k in ("score", "spx", "vix"):
        a, b = cur["market"].get(k), prev["market"].get(k)
        if a is not None and b is not None:
            market[k] = {"from": b, "to": a}
    return {"base": prev["date"], "flag_changes": flag_changes, "macro_changes": macro_changes,
            "checklist_changes": checklist_changes, "market": market}
