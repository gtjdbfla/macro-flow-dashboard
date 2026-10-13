"""feature 빌드 진입점: data/parquet/features_*.parquet + data/features_meta.json 생성."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build import build_all, save  # noqa: E402

out = build_all()
meta = save(out)
for name, df in out.items():
    valid = df.notna().any(axis=1)
    print(f"{name:10s} cols={df.shape[1]:4d} rows={len(df)} first_row={valid.idxmax().date() if valid.any() else None} "
          f"last_nonnull={df.dropna(how='all').index[-1].date()}")
st = out["state"].iloc[-1]
print("현재 상태:", {k: st[k] for k in ["growth", "inflation", "regime", "policy", "liquidity", "dollar"]})
