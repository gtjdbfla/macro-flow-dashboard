"""AI 수익성 모듈 계산 -> data/results/ai_*.  (data/parquet/sec_quarterly.parquet 필요: collect.py sec)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from backtest import ai_capex_event  # noqa: E402
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
sec = pd.read_parquet(PARQUET / "sec_quarterly.parquet")
(out / "ai_flags.json").write_text(json.dumps({"flags": fl, "latest_filed": str(sec["filed"].max().date())},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
ev = ai_capex_event.run()
ev.to_parquet(out / "ai_event_study.parquet")
print("flags:", [(f["id"], f["level"]) for f in fl])
print("event rows:", len(ev))
