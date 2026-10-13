"""전 레이어 feature 빌드. build_all(asof=T)는 T 이후 정보를 쓰지 않는다."""
import json

import pandas as pd

from src.data import DATA_DIR, PARQUET

from . import (flow_features, korea_features, liquidity_features, macro_features, regime, risk_features,
               sector_features)
from .common import Ctx

LAYERS = {"macro": macro_features, "liquidity": liquidity_features, "risk": risk_features,
          "sector": sector_features, "flow": flow_features, "korea": korea_features}


def build_all(asof=None) -> dict:
    ctx = Ctx(asof)
    out = {name: mod.build(ctx).reindex(ctx.cal) for name, mod in LAYERS.items()}
    out["state"] = regime.build(out["macro"], out["liquidity"])
    return out


def save(out: dict):
    meta = {}
    for name, df in out.items():
        df = df.copy()
        num = df.select_dtypes("number").columns
        df[num] = df[num].astype("float32")
        df.to_parquet(PARQUET / f"features_{name}.parquet")
        meta[name] = {c: (None if df[c].first_valid_index() is None else str(df[c].first_valid_index().date()))
                      for c in df.columns}
    (DATA_DIR / "features_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta
