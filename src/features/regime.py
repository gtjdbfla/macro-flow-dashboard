"""Macro Regime 분류 v2 — 규칙 고정, 수익률 기준 튜닝 없음.

v1은 입력을 매일 부호/3분위로 잘라 라벨이 연 14~32회 뒤집혔다(중앙값 지속 3~7일; 월간 지표 중심의 '국면'으로는 너무 잦음).
v2는 구조적 이유만으로 두 가지를 더했다: ① 입력을 21거래일(≈월 1회 갱신 주기) 평균으로 평활, ② 슈미트 트리거(진입/이탈
임계가 다른 히스테리시스). 임계값은 3분위 경계(진입)와 중앙값(이탈), z의 ±0.15로 정했고 **수익률을 보고 고르지 않았다**(지속성만 확인).

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


SMOOTH = 21
Z_BAND = 0.15


def tercile(rank: pd.Series, hi_label: str, lo_label: str) -> pd.Series:
    """3상태 슈미트 트리거: 순위 >2/3면 hi 진입, <1/2면 이탈 / <1/3면 lo 진입, >1/2면 이탈. 입력은 21일 평균."""
    r = rank.rolling(SMOOTH, min_periods=SMOOTH // 2).mean()
    state, out = "Neutral", []
    for v in r.to_numpy():
        if v != v:
            out.append(None)
            continue
        if state == "Neutral":
            state = hi_label if v > 2 / 3 else (lo_label if v < 1 / 3 else "Neutral")
        elif state == hi_label and v < 0.5:
            state = "Neutral"
        elif state == lo_label and v > 0.5:
            state = "Neutral"
        out.append(state)
    return pd.Series(out, index=rank.index, dtype="object")


def two_state(z: pd.Series, up: str, down: str) -> pd.Series:
    """2상태 슈미트 트리거: +band 위로 가야 up, -band 아래로 가야 down. 입력은 21일 평균."""
    sm = z.rolling(SMOOTH, min_periods=SMOOTH // 2).mean()
    state, out = None, []
    for v in sm.to_numpy():
        if v != v:
            out.append(state)  # 입력이 끊기면 마지막 상태 유지(초기 구간은 None)
            continue
        if state is None:
            state = up if v > 0 else down
        elif state == down and v > Z_BAND:
            state = up
        elif state == up and v < -Z_BAND:
            state = down
        out.append(state)
    return pd.Series(out, index=z.index, dtype="object")


def _policy(chg: pd.Series) -> pd.Series:
    """정책금리 6개월 변화: ±25bp 진입, ±10bp 안으로 들어와야 Neutral 복귀(히스테리시스)."""
    state, out = "Neutral", []
    for v in chg.to_numpy():
        if v != v:
            out.append(None)
            continue
        if state == "Neutral":
            state = "Tightening" if v > 0.25 else ("Easing" if v < -0.25 else "Neutral")
        elif state == "Tightening" and v < 0.10:
            state = "Neutral"
        elif state == "Easing" and v > -0.10:
            state = "Neutral"
        out.append(state)
    return pd.Series(out, index=chg.index, dtype="object")


def build(macro: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    g = _mean_z(macro, GROWTH_IN)
    i = _mean_z(macro, INFL_IN)
    ff = macro.get("macro__FedFunds__level")
    pol_chg = ff.diff(126) if ff is not None else pd.Series(np.nan, index=macro.index)
    st = pd.DataFrame(index=macro.index)
    st["growth_z"], st["inflation_z"] = g, i
    st["growth"] = two_state(g, "Accelerating", "Decelerating").where(g.rolling(SMOOTH, min_periods=1).apply(
        lambda x: np.isfinite(x).any(), raw=True) > 0)
    st["inflation"] = two_state(i, "Accelerating", "Decelerating").where(i.rolling(SMOOTH, min_periods=1).apply(
        lambda x: np.isfinite(x).any(), raw=True) > 0)
    reg = {("Accelerating", "Decelerating"): "Goldilocks", ("Accelerating", "Accelerating"): "Reflation",
           ("Decelerating", "Decelerating"): "Slowdown", ("Decelerating", "Accelerating"): "Stagflation"}
    st["regime"] = [reg.get((a, b)) for a, b in zip(st["growth"], st["inflation"])]
    st["policy_chg_6m"] = pol_chg
    st["policy"] = _policy(pol_chg)
    r = liq.get("liq__NetLiquidity__chg_60d_rank")
    st["liquidity"] = tercile(r, "Expanding", "Contracting") if r is not None else None
    d = macro.get("macro__DXY__chg_60d_rank")
    st["dollar"] = tercile(d, "Strong", "Weak") if d is not None else None
    cols = ["regime", "policy", "liquidity", "dollar"]
    st["state_key"] = st[cols].apply(lambda r: "|".join(map(str, r)) if r.notna().all() else None, axis=1)
    return st
