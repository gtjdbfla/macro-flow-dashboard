"""투자 길잡이 스냅샷 -> data/ledger/guide.jsonl(날짜별 누적)과 data/results/ai_guide.json(최신 상태와 지난주 대비 변화)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src.ai import early_warning as EW  # noqa: E402
from src.ai import guide as G  # noqa: E402
from src.data import DATA_DIR  # noqa: E402

R = DATA_DIR / "results"
risk = json.loads((R / "risk_gauge.json").read_text(encoding="utf-8"))
flags = json.loads((R / "ai_flags.json").read_text(encoding="utf-8"))["flags"]
big4 = pd.read_parquet(R / "ai_history.parquet").query("ticker == 'BIG4'")
latest = pd.read_parquet(R / "ai_latest.parquet").set_index("ticker")
b = latest.loc["BIG4"]
metrics = {"big4_capex_yoy": float(b.capex_yoy), "big4_capex_ocf": float(b.capex_ocf), "big4_fcf_ttm": float(b.fcf_ttm),
           "big4_fcf_chg": float(b.fcf_chg), "big4_rev_yoy": float(b.rev_yoy), "big4_opinc_yoy": float(b.opinc_yoy)}


def _fred(sid):
    return pd.read_parquet(DATA_DIR / "parquet" / f"fred_{sid}.parquet")["value"].dropna()


def _ctx():
    """거품론 점검표 입력. 못 읽은 값은 None으로 둬서 해당 항목이 '자료 부족'이 되게 한다."""
    from src.ai import end_demand as ED
    ctx = {}
    try:
        rr = _fred("DFII10"); ctx["real_rate_3m"] = float(rr.iloc[-1] - rr.iloc[-64])
        hy = _fred("BAMLH0A0HYM2"); ctx["hy_4w"] = float(hy.iloc[-1] - hy.iloc[-21])
    except Exception as e:
        print("guide: FRED 입력 실패", e)
    try:
        dq = pd.read_parquet(R / "ai_demand_quality.parquet").set_index("period_end").support_ratio.dropna()
        ctx["support_ratio"] = [float(v) for v in dq.tail(2)]
        cov = pd.read_parquet(R / "ai_end_revenue_coverage.parquet").dropna(subset=["coverage"]).query("ticker == 'MSFT'").sort_values("period_end")
        ctx["coverage"] = [(str(r.period_end.date()), float(r.coverage)) for r in cov.tail(2).itertuples()]
        cgn = pd.read_parquet(R / "ai_cloud_gain.parquet").sort_values("quarter_end")
        ctx["cloud_gain"] = {t: [(str(r.quarter_end.date()), float(r.ratio)) for r in g.dropna(subset=["ratio"]).itertuples()] for t, g in cgn.groupby("ticker")}
        aq = ED.anthropic_quarters()
        ctx["anthropic_ratio"] = [float(v) for v in aq.compute_ratio.tail(2)]
        cds = sorted((d, v) for d, v, t, _ in ED.CRWV_STRESS if "CDS" in t)
        ctx["crwv_cds"] = [(d, v) for d, v in cds]
    except Exception as e:
        print("guide: 입력 일부 실패", e)
    try:
        from src.data import PARQUET
        from src.data.yahoo import _fname
        soxx = pd.read_parquet(PARQUET / _fname("SOXX"))["close"].dropna()
        ctx["soxx_1m"] = float(soxx.iloc[-1] / soxx.iloc[-22] - 1)
        ctx["vix"] = risk.get("vix")
    except Exception as e:
        print("guide: 시장 입력 실패", e)
    return ctx


(DATA_DIR / "ledger").mkdir(exist_ok=True)
snap = G.snapshot(pd.Timestamp.now().date(), risk, flags, big4, metrics, _ctx())
try:
    ew = EW.compute()
    snap["early"] = EW.summary(ew)
    _prev = [json.loads(x) for x in (DATA_DIR / "ledger" / "guide.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] if (DATA_DIR / "ledger" / "guide.jsonl").exists() else []
    _prev = [r for r in _prev if r["date"] != snap["date"] and r.get("early", {}).get("stage")]
    if _prev and _prev[-1]["early"]["stage"] != snap["early"]["stage"]:
        import os
        import requests
        msg = f"시장 단계 변경: {_prev[-1]['early']['stage']} → {ew['stage']['name']} ({ew['stage']['why']})"
        print("guide:", msg)
        if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
            try:
                requests.post(f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
                              data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": msg}, timeout=15)
            except Exception as e:
                print("guide: 알림 실패", e)
    (R / "ai_early_warning.json").write_text(json.dumps(ew, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (R / "ai_early_warning.json").chmod(0o644)
    print("guide: 시장 단계", ew["stage"]["name"], "|", ew["stage"]["why"], "| 조기경보 빨강", ew["red"], "노랑", ew["yellow"], "초록", ew["green"], "| 이탈", [t["ticker"] for t in ew["trend"] if t["state"] == "이탈"])
except Exception as e:
    print("guide: 조기경보 계산 실패", e)
rows = G.append(DATA_DIR / "ledger" / "guide.jsonl", snap)
out = {"snapshot": snap, "changes": G.compare(rows), "snapshots": len(rows)}
try:
    from src.ai import scorecard as SC
    from src.data import PARQUET
    from src.data.yahoo import _fname
    qqq = pd.read_parquet(PARQUET / _fname("QQQ"))["close"].dropna()
    sc = SC.score(rows, qqq)
    (R / "ai_scorecard.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1), encoding="utf-8")
    (R / "ai_scorecard.json").chmod(0o644)
    print("guide: 전향 성적표 스냅샷", sc["snapshots"], "확정", sc["matured"])
except Exception as e:
    print("guide: 성적표 실패", e)
tmp = R / "ai_guide.json.tmp"
tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
tmp.chmod(0o644)
tmp.replace(R / "ai_guide.json")
b = snap["bubble"]
print("guide: 거품론 해소 조건 예", [i["ok"] for i in b["resolution"]].count(True), "/", len(b["resolution"]),
      "| 악화 신호 발생", [i["ok"] for i in b["warning"]].count(True), "/", len(b["warning"]))
ok = [i["ok"] for i in snap["checklist"]]
print("guide:", snap["date"], "점검표 예", ok.count(True), "/ 아니오", ok.count(False), "/ 자료 부족", ok.count(None), "| 스냅샷", len(rows))
