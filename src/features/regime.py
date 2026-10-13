"""Macro Regime 분류 — 규칙 고정, 튜닝 없음.

Growth/Inflation: 선행 구성요소의 '3개월 yoy 변화'를 expanding z로 표준화해 평균, 부호로 가속/감속.
Monetary: 정책금리 6개월(126일) 변화 ±25bp.   Liquidity/Dollar: 60일 변화의 expanding 3분위(과거 분포 기준).
각 라벨은 필요한 입력이 없으면 NaN(알 수 없음)으로 둔다."""
import numpy as np
import pandas as pd

GROWTH_IN = ["macro__IndProd__yoy_chg3m_z", "macro__Payrolls__yoy_chg3m_z", "macro__RetailSales__yoy_chg3m_z"]
INFL_IN = ["macro__CoreCPI__yoy_chg3m_z", "macro__CorePCE__yoy_chg3m_z", "macro__BEI10Y__chg_60d_z"]


def _mean_z(df, cols):
    have = [c for c in cols if c in df]
    if not have:
        return pd.Series(np.nan, index=df.index)
    sub = df[have]
    return sub.mean(axis=1, skipna=True).where(sub.notna().sum(axis=1) >= max(1, len(cols) - 1))


def tercile(rank: pd.Series, hi_label: str, lo_label: str) -> pd.Series:
    out = pd.Series("Neutral", index=rank.index, dtype="object")
    out[rank > 2 / 3] = hi_label
    out[rank < 1 / 3] = lo_label
    out[rank.isna()] = None
    return out


def build(macro: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    g = _mean_z(macro, GROWTH_IN)
    i = _mean_z(macro, INFL_IN)
    ff = macro.get("macro__FedFunds__level")
    pol_chg = ff.diff(126) if ff is not None else pd.Series(np.nan, index=macro.index)
    st = pd.DataFrame(index=macro.index)
    st["growth_z"], st["inflation_z"] = g, i
    st["growth"] = np.where(g.isna(), None, np.where(g > 0, "Accelerating", "Decelerating"))
    st["inflation"] = np.where(i.isna(), None, np.where(i > 0, "Accelerating", "Decelerating"))
    reg = {("Accelerating", "Decelerating"): "Goldilocks", ("Accelerating", "Accelerating"): "Reflation",
           ("Decelerating", "Decelerating"): "Slowdown", ("Decelerating", "Accelerating"): "Stagflation"}
    st["regime"] = [reg.get((a, b)) for a, b in zip(st["growth"], st["inflation"])]
    st["policy_chg_6m"] = pol_chg
    st["policy"] = np.where(pol_chg.isna(), None, np.where(pol_chg > 0.25, "Tightening",
                                                           np.where(pol_chg < -0.25, "Easing", "Neutral")))
    r = liq.get("liq__NetLiquidity__chg_60d_rank")
    st["liquidity"] = tercile(r, "Expanding", "Contracting") if r is not None else None
    d = macro.get("macro__DXY__chg_60d_rank")
    st["dollar"] = tercile(d, "Strong", "Weak") if d is not None else None
    cols = ["regime", "policy", "liquidity", "dollar"]
    st["state_key"] = st[cols].apply(lambda r: "|".join(map(str, r)) if r.notna().all() else None, axis=1)
    return st
