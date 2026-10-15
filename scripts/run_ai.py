"""AI 수익성 모듈 계산 -> data/results/ai_*.  (data/parquet/sec_quarterly.parquet 필요: collect.py sec)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from backtest import ai_capex_event  # noqa: E402
from src.ai import cloud as C  # noqa: E402
from src.ai import forecast as FC  # noqa: E402
from src.ai import safety as SF  # noqa: E402
from src.ai import credit as CR  # noqa: E402
from src.ai import metrics as M  # noqa: E402
from src.data import DATA_DIR, PARQUET  # noqa: E402
from src.data.yahoo import _fname  # noqa: E402

out = DATA_DIR / "results"
out.mkdir(exist_ok=True)
co = M.all_companies()
M.latest_table(co).to_parquet(out / "ai_latest.parquet")
hist = []
for t, (o, a) in co.items():
    h = o.copy()
    h["ticker"], h["available"] = t, a.reindex(h.index)
    hist.append(h.reset_index(names="period_end"))
pd.concat(hist).to_parquet(out / "ai_history.parquet")

prices = {}
for sym, name in [("SOXX", "SOXX"), ("NVDA", "NVDA"), ("000660.KS", "SK하이닉스")]:
    f = PARQUET / _fname(sym)
    if f.exists():
        prices[name] = pd.read_parquet(f)["adjclose"].dropna()
fl = M.flags(co, prices)
ch = C.history()
ch.to_parquet(out / "ai_cloud_history.parquet")
C.latest(ch, co).to_parquet(out / "ai_cloud_latest.parquet")
C.aggregate(ch).reset_index().to_parquet(out / "ai_cloud_agg.parquet")
tw_f = PARQUET / "twse_monthly.parquet"
tw = pd.read_parquet(tw_f) if tw_f.exists() else None
dbt_f = PARQUET / "debt_issuance.parquet"
cr_fl = []
if dbt_f.exists():
    dd = CR.issuance()
    CR.spread_series(dd).to_parquet(out / "ai_credit_spreads.parquet")
    dd.to_parquet(out / "ai_credit_issuance.parquet")
    pr2 = dict(prices)
    f_ = PARQUET / _fname("CRWV")
    if f_.exists():
        pr2["CRWV"] = pd.read_parquet(f_)["adjclose"].dropna()
    cr_fl = CR.flags(dd, pr2)
try:
    s_lt, s_mh, s_val, s_ro, s_rc, s_oas, s_fl, s_mat, s_valn = SF.run()
    s_lt.to_parquet(out / "ai_safety_latest.parquet")
    s_mh.to_parquet(out / "ai_merton_history.parquet")
    s_val.to_parquet(out / "ai_merton_vs_issued.parquet")
    s_ro.to_parquet(out / "ai_roic_big4.parquet")
    s_rc.to_parquet(out / "ai_roic_company.parquet")
    s_oas.to_parquet(out / "ai_credit_oas.parquet")
    s_mat.to_parquet(out / "ai_debt_maturity.parquet")
    s_valn.to_parquet(out / "ai_valuation.parquet")
except Exception as e:  # 안전성 지표 실패가 나머지 AI 계산을 막지 않게 한다
    print("safety 실패:", e)
    s_fl = []
fl = sorted(fl + C.flags(ch, tw) + cr_fl + s_fl, key=lambda f: int(f["id"][1:]))
sec = pd.read_parquet(PARQUET / "sec_quarterly.parquet")
(out / "ai_flags.json").write_text(json.dumps({"flags": fl, "latest_filed": str(sec["filed"].max().date())},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
ev = ai_capex_event.run()
ev.to_parquet(out / "ai_event_study.parquet")
fe, fp, fn = FC.run()
fe.to_parquet(out / "ai_forecast_eval.parquet")
fp.to_parquet(out / "ai_forecast.parquet")
fn.to_parquet(out / "ai_forecast_negatives.parquet")
print("forecast:", fp[["metric", "now_ttm", "proj", "growth"]].round(2).values.tolist())
print("flags:", [(f["id"], f["level"]) for f in fl])
print("event rows:", len(ev))
