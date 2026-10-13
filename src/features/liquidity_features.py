"""Layer 2 — Liquidity. 순유동성 = Fed 대차대조표 - TGA - 역RP (십억달러).
H.4.1은 수요일 기준 목요일 공표 -> +1일. NFCI는 금요일 기준 다음 수요일 공표 -> +5일."""
import pandas as pd

from .common import Ctx, exp_rank, exp_z, feats_series, place


def build(ctx: Ctx) -> pd.DataFrame:
    walcl = ctx.fred_on_cal("WALCL", 1)   # 백만달러
    tga = ctx.fred_on_cal("WTREGEN", 1) / 1000  # 백만달러 -> 십억달러
    rrp = ctx.fred_on_cal("RRPONTSYD", 1)  # 십억달러(2013 이전 없음 -> 0)
    parts = []
    if walcl is not None and tga is not None:
        rrp = rrp.fillna(0) if rrp is not None else 0
        net = (walcl / 1000 - tga - rrp).dropna()
        # 이미 달력에 올라간 값이므로 달력 행 기준 변화(≈거래일)
        F = feats_series(net, "NetLiquidity", "diff", hs=(5, 20, 60, 120), unit="d", key_h=60)
        F["NetLiquidity__chg_60d_pct"] = net.pct_change(60)
        parts.append(F.reindex(ctx.cal))
        parts.append(feats_series(walcl.dropna() / 1000, "FedBalanceSheet", "diff",
                                  hs=(5, 20, 60), with_rank=False).reindex(ctx.cal))
        parts.append(feats_series(tga.dropna(), "TGA", "diff", hs=(5, 20, 60), with_rank=False).reindex(ctx.cal))
    nfci, avail = ctx.fred_daily_native("NFCI", 5)
    if nfci is not None:
        parts.append(place(feats_series(nfci, "NFCI", "diff", hs=(1, 4, 13), unit="w", minp=260),
                           avail, ctx.cal))
    broad, avail = ctx.fred_daily_native("DTWEXBGS", 1)
    if broad is not None:
        parts.append(place(feats_series(broad, "BroadUSD", "pct"), avail, ctx.cal))
    return pd.concat(parts, axis=1).add_prefix("liq__")
