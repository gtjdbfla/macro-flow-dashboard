"""Layer 1 — Macro. 월/분기 지표는 최초 공표값(first print)과 공표시점으로 배치한다."""
import pandas as pd

from .common import MINP_M, Ctx, exp_rank, exp_z, feats_series, place

# 일별 시장 지표: (FRED id, 이름, kind, 발표지연일)
DAILY = [
    ("DFF", "FedFunds", "diff", 1), ("DGS2", "UST2Y", "diff", 1), ("DGS10", "UST10Y", "diff", 1),
    ("DFII10", "Real10Y", "diff", 1), ("T10YIE", "BEI10Y", "diff", 1), ("T10Y2Y", "Curve10s2s", "diff", 1),
    ("DCOILWTICO", "WTI", "pct", 1),
]
# 월/분기: (id, 이름, 기본지연일, 종류)  종류 idx=지수(yoy/3m연율), rate=비율(수준/차분), gdp=분기, claims=주간
MONTHLY = [
    ("CPIAUCSL", "CPI", 45, "idx"), ("CPILFESL", "CoreCPI", 45, "idx"),
    ("PCEPI", "PCE", 60, "idx"), ("PCEPILFE", "CorePCE", 60, "idx"),
    ("PAYEMS", "Payrolls", 37, "idx"), ("INDPRO", "IndProd", 47, "idx"),
    ("RSAFS", "RetailSales", 45, "idx"), ("M2SL", "M2", 50, "idx"), ("DGORDER", "DurableOrders", 57, "idx"),
    ("UNRATE", "Unemployment", 37, "rate"), ("GDPC1", "RealGDP", 120, "gdp"), ("ICSA", "Claims", 5, "claims"),
]


def _monthly_feats(fp: pd.DataFrame, name: str, kind: str) -> pd.DataFrame:
    s = fp["value"]
    o = {}
    if kind == "idx":
        yoy = s.pct_change(12)
        o[f"{name}__yoy"] = yoy
        o[f"{name}__ann3m"] = (s / s.shift(3)) ** 4 - 1
        o[f"{name}__yoy_chg3m"] = yoy.diff(3)
        o[f"{name}__yoy_chg3m_z"] = exp_z(yoy.diff(3), MINP_M)
        o[f"{name}__yoy_rank"] = exp_rank(yoy, MINP_M)
    elif kind == "rate":
        o[f"{name}__level"] = s
        o[f"{name}__chg_3m"] = s.diff(3)
        o[f"{name}__chg_12m"] = s.diff(12)
        o[f"{name}__level_rank"] = exp_rank(s, MINP_M)
    elif kind == "gdp":
        o[f"{name}__qoq_ann"] = (s / s.shift(1)) ** 4 - 1
        o[f"{name}__yoy"] = s.pct_change(4)
    else:  # claims(주간): 4주 평균과 13주 변화
        ma = s.rolling(4).mean()
        o[f"{name}__ma4w"] = ma
        o[f"{name}__ma4w_chg13w"] = ma.pct_change(13)
        o[f"{name}__ma4w_rank"] = exp_rank(ma, 104)
    return pd.DataFrame(o)


def build(ctx: Ctx) -> pd.DataFrame:
    parts = []
    for sid, name, kind, lag in DAILY:
        s, avail = ctx.fred_daily_native(sid, lag)
        if s is None:
            continue
        parts.append(place(feats_series(s, name, kind), avail, ctx.cal))
    # 달러(DXY): 가격 시계열
    dxy = ctx.close("DX-Y.NYB", adj=False)
    if dxy is not None:
        parts.append(feats_series(dxy.dropna(), "DXY", "pct").reindex(ctx.cal))
    for sid, name, lag, kind in MONTHLY:
        fp = ctx.first_print(sid, lag, max_lag=lag * 3)
        if fp is None or len(fp) < 14:
            continue
        F = _monthly_feats(fp, name, kind)
        parts.append(place(F, fp["available_at"], ctx.cal))
    return pd.concat(parts, axis=1).add_prefix("macro__")
