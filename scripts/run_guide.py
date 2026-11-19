"""투자 길잡이 스냅샷 -> data/ledger/guide.jsonl(날짜별 누적)과 data/results/ai_guide.json(최신 상태와 지난주 대비 변화)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

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
(DATA_DIR / "ledger").mkdir(exist_ok=True)
snap = G.snapshot(pd.Timestamp.now().date(), risk, flags, big4, metrics)
rows = G.append(DATA_DIR / "ledger" / "guide.jsonl", snap)
out = {"snapshot": snap, "changes": G.compare(rows), "snapshots": len(rows)}
tmp = R / "ai_guide.json.tmp"
tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
tmp.chmod(0o644)
tmp.replace(R / "ai_guide.json")
ok = [i["ok"] for i in snap["checklist"]]
print("guide:", snap["date"], "점검표 예", ok.count(True), "/ 아니오", ok.count(False), "/ 자료 부족", ok.count(None), "| 스냅샷", len(rows))
