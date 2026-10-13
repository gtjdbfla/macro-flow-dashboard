"""Layer 4 — Sector/Industry. 절대수익률, SPY 대비 상대강도(RS), 모멘텀, 거래량, 섹터 breadth.
가격 수준은 쓰지 않는다(조정종가가 사후 배당으로 소급 조정되므로 수익률·비율 변화만 사용)."""
import numpy as np
import pandas as pd

from .common import Ctx, exp_z

SECTORS = {"XLK": "Technology", "XLC": "Communication", "XLY": "ConsDisc", "XLP": "ConsStaples",
           "XLF": "Financials", "XLV": "Healthcare", "XLI": "Industrials", "XLE": "Energy",
           "XLB": "Materials", "XLU": "Utilities", "XLRE": "RealEstate"}
INDUSTRIES = {"SOXX": "Semis", "SMH": "SemisVanEck", "XSD": "SemisEqual", "IGV": "Software",
              "CIBR": "Cyber", "KBE": "Banks", "KIE": "Insurance", "IBB": "Biotech", "XPH": "Pharma",
              "IHI": "MedDevices", "ITA": "AeroDefense", "XOP": "OilGas", "OIH": "EnergyServices",
              "URA": "Nuclear", "COPX": "Copper", "XME": "MetalsMining", "SLX": "Steel"}
HS = (1, 3, 5, 10, 20, 60, 120)


def _one(ctx, sym, label, spy):
    c = ctx.close(sym)
    if c is None:
        return None
    lc = np.log(c)
    ls = np.log(spy)
    o = {}
    for h in HS:
        o[f"{label}__ret_{h}d"] = c.pct_change(h, fill_method=None)
        o[f"{label}__rs_{h}d"] = (lc - ls).diff(h)  # SPY 대비 초과로그수익
    o[f"{label}__mom_12_1"] = c.shift(21) / c.shift(252) - 1
    o[f"{label}__rs_accel"] = o[f"{label}__rs_5d"] - o[f"{label}__rs_20d"] * (5 / 20)
    v = ctx.volume(sym)
    if v is not None:
        o[f"{label}__volume_z"] = exp_z(np.log(v.rolling(20).mean() / v.rolling(252).mean()))
    return pd.DataFrame(o)


def build(ctx: Ctx) -> pd.DataFrame:
    spy = ctx.close("SPY")
    parts = []
    for sym, label in {**SECTORS, **INDUSTRIES}.items():
        f = _one(ctx, sym, label, spy)
        if f is not None:
            parts.append(f)
    # breadth: 11개 섹터 ETF(해당일 존재하는 것만 분모)
    ma50, ma200, pos, n = {}, {}, {}, {}
    for sym in SECTORS:
        c = ctx.close(sym)
        if c is None:
            continue
        first = c.first_valid_index()
        live = c.index >= first
        ma50[sym] = (c > c.rolling(50).mean()).where(live & c.rolling(50).mean().notna())
        ma200[sym] = (c > c.rolling(200).mean()).where(live & c.rolling(200).mean().notna())
        pos[sym] = ((np.log(c) - np.log(spy)).diff(20) > 0).where(live)
    b = pd.DataFrame({
        "breadth__sectors_above_50dma": pd.DataFrame(ma50).mean(axis=1, skipna=True),
        "breadth__sectors_above_200dma": pd.DataFrame(ma200).mean(axis=1, skipna=True),
        "breadth__sectors_outperf_spy_20d": pd.DataFrame(pos).mean(axis=1, skipna=True),
        "breadth__n_sectors_live": pd.DataFrame(pos).notna().sum(axis=1),
    })
    return pd.concat(parts + [b], axis=1).add_prefix("sector__")
