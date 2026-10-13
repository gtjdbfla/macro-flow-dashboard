"""섹터/산업 회전 지도. 가격 점수 = SPY 대비 상대강도(5·20·60D) 섹터 간 순위, 자금 점수 = SPY 대비 체이킨 자금흐름(CMF 5·20·60D) 순위, 합성 = 동일가중.
주의: 백테스트에서 이 feature들의 예측력은 분할 간 불안정했다(docs/BACKTEST.md) -> '현재 스냅샷'이지 예측이 아니다."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.features.sector_features import INDUSTRIES, SECTORS


def arrows(rs3, rs5, rs20):
    pos = int(rs3 > 0) + int(rs5 > 0) + int(rs20 > 0)
    if abs(rs5) < 0.0025 and abs(rs3) < 0.0015:
        return "→"
    return {3: "↑↑", 2: "↑", 1: "↓", 0: "↓↓"}[pos]


def snapshot(group: dict, row: pd.Series, ret: pd.Series) -> pd.DataFrame:
    recs = []
    for sym, lab in group.items():
        g = lambda f: row.get(f"sector__{lab}__{f}", np.nan)  # noqa: E731
        if np.isnan(g("rs_20d")):
            continue
        recs.append({"symbol": sym, "name": lab, "rs_1m": g("rs_20d"), "rs_1w": g("rs_5d"), "rs_3d": g("rs_3d"),
                     "rs_3m": g("rs_60d"), "ret_1m": row.get(f"sector__{lab}__ret_20d", np.nan),
                     "vol_z": g("volume_z"), "accel": g("rs_accel"),
                     "cmf_3d": g("cmf_rel_3d"), "cmf_1w": g("cmf_rel_5d"), "cmf_1m": g("cmf_rel_20d"),
                     "cmf_3m": g("cmf_rel_60d")})
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    d["score_price"] = (100 * d[["rs_1w", "rs_1m", "rs_3m"]].rank(pct=True).mean(axis=1)).round(0)
    d["score_flow"] = (100 * d[["cmf_1w", "cmf_1m", "cmf_3m"]].rank(pct=True).mean(axis=1)).round(0)
    d["score"] = ((d.score_price + d.score_flow) / 2).round(0)  # 가격(RS)과 자금(CMF) 동일가중
    d["direction"] = [arrows(a, b, c) for a, b, c in zip(d.rs_3d, d.rs_1w, d.rs_1m)]
    d["flow_dir"] = [arrows(a, b, c) for a, b, c in zip(d.cmf_3d, d.cmf_1w, d.cmf_1m)]
    return d.sort_values("score", ascending=False).reset_index(drop=True)


def build(asof=None):
    S = pd.read_parquet(PARQUET / "features_sector.parquet")
    row = S.loc[asof] if asof is not None else S.dropna(how="all").iloc[-1]
    return snapshot(SECTORS, row, None), snapshot(INDUSTRIES, row, None), row.name
