"""Layer 4b — Institutional positioning (CFTC TFF). 확보 가능한 포지셔닝만 사용한다.
ETF Fund Flow / 옵션 / 13F는 무료 장기 이력이 없어 V1에서 제외(docs/DESIGN.md).
보고 기준일=화요일, 공표=금요일 -> available_at = 기준일+3일(수집 단계에서 기록)."""
import pandas as pd

from src.data import PARQUET
from src.data.universe import CFTC

from .common import Ctx, exp_rank, place


def build(ctx: Ctx) -> pd.DataFrame:
    f = PARQUET / "cftc_tff.parquet"
    if not f.exists():
        return pd.DataFrame(index=ctx.cal)
    df = pd.read_parquet(f)
    if ctx.asof is not None:
        df = df[df["available_at"] <= ctx.asof]
    parts = []
    for code, label in CFTC.items():
        g = df[df["cftc_contract_market_code"] == code].sort_values("date").set_index("date")
        if len(g) < 60:
            continue
        oi = g["open_interest_all"].where(g["open_interest_all"] > 0)
        o = {
            "levmoney_net": (g["lev_money_positions_long"] - g["lev_money_positions_short"]) / oi,
            "assetmgr_net": (g["asset_mgr_positions_long"] - g["asset_mgr_positions_short"]) / oi,
            "dealer_net": (g["dealer_positions_long_all"] - g["dealer_positions_short_all"]) / oi,
        }
        F = {}
        key = label.replace(" ", "_").replace("&", "").replace("/", "_")
        for k, s in o.items():
            F[f"{key}__{k}"] = s
            F[f"{key}__{k}_chg4w"] = s.diff(4)
            F[f"{key}__{k}_rank"] = exp_rank(s, 104)
        parts.append(place(pd.DataFrame(F), g["available_at"], ctx.cal))
    return pd.concat(parts, axis=1).add_prefix("cftc__")
