"""위험 온도계 계산 -> data/results/risk_gauge*.  (data/parquet/px_*.parquet 필요: collect.py yahoo)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring import risk_gauge as RG  # noqa: E402
from src.data import DATA_DIR  # noqa: E402

out = DATA_DIR / "results"
out.mkdir(exist_ok=True)
hist, by, kospi, now = RG.run()
hist.loc["2005-01-01":].iloc[::5].to_parquet(out / "risk_gauge_history.parquet")
by.to_parquet(out / "risk_gauge_levels.parquet")
kospi.to_parquet(out / "risk_gauge_kospi.parquet")
(out / "risk_gauge.json").write_text(json.dumps(now, ensure_ascii=False, indent=1), encoding="utf-8")
print("risk:", now["as_of"], round(now["score"], 2), now["level"], "shock" if now["shock"] else "", "| KOSPI dd", now["kospi_dd"])
print(by.round(3).to_string(index=False))
