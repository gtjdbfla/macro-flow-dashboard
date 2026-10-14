"""전체 백테스트 실행 -> data/results/*.parquet. (features가 먼저 빌드돼 있어야 함)"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest import flow_rules, korea_sector_flow, korea_transmission, lead_lag, regime_backtest, sector_rotation  # noqa: E402
from src.data import DATA_DIR  # noqa: E402

out = DATA_DIR / "results"
out.mkdir(exist_ok=True)
t0 = time.time()
rb = regime_backtest.run()
rb.to_parquet(out / "regime_backtest.parquet")
rep = regime_backtest.replication(rb)
rep.to_parquet(out / "regime_replication.parquet")
t = rep[rep.testable]
print(f"regime: rows={len(rb)} testable={len(t)} train_sig={int(t.train_sig.sum())} "
      f"replicated={int(t.replicated.sum())} ({t.replicated.mean():.2%}, 귀무 기대 약 1.25%)")
sector_rotation.run().to_parquet(out / "sector_rotation.parquet")
lead_lag.run().to_parquet(out / "lead_lag.parquet")
korea_transmission.run().to_parquet(out / "korea_transmission.parquet")
korea_sector_flow.run().to_parquet(out / "korea_sector_flow.parquet")
flow_rules.run().to_parquet(out / "flow_rules.parquet")
print(f"done {time.time() - t0:.0f}s ->", sorted(p.name for p in out.glob('*.parquet')))
