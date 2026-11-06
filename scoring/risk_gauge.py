"""위험 온도계 — '지금 큰 낙폭이 올 위험이 높은 상태인가'. 투자 지시가 아니라 상태 표시다.

설계 근거(2026-11-06 연구, docs/RISK_GAUGE.md)
  * 타깃: S&P500이 앞으로 126거래일(6개월) 안에 오늘 종가 대비 -15% 이상 빠지는가(전체 기간 11.8%).
  * 지표 5개(전부 '클수록 위험', 방향은 사전 고정): 200일선 이탈, 12개월 모멘텀 약세, 1년 고점 대비 낙폭, 3개월 실현변동성, 달러 6개월 강세.
    1967~2006 두 구간 모두 AUC>0.55였던 것만 채택. 금리 인상 속도·장단기 금리차·CAPE는 시대마다 방향이 뒤집히거나 효과가 없어 제외.
  * 각 지표는 과거만 쓴 확장 백분위(최소 5년), 점수 = 5개 평균. 구간(낮음<0.4·보통<0.6·높음<0.8·매우 높음)은 사전 고정.
  * 한계: 평균 수익 방향은 알려 주지 못한다(점수가 높아도 앞 6개월 평균 수익은 플러스). 표본외 AUC 약 0.56."""
import bisect

import numpy as np
import pandas as pd

from src.data import PARQUET
from src.data.yahoo import _fname

H = 126
BAD = -0.15
START = "1962-01-02"
LEVELS = [(0.0, "낮음"), (0.4, "보통"), (0.6, "높음"), (0.8, "매우 높음")]
PARTS = {"below_200dma": "200일선 이탈", "neg_mom_12m": "12개월 모멘텀 약세", "drawdown_1y": "1년 고점 대비 낙폭",
         "realized_vol_3m": "3개월 변동성", "dollar_up_6m": "달러 6개월 강세"}
SHOCK_RET5, SHOCK_VIX5 = -0.05, 0.60   # 충격 표지(사전 고정): S&P 5일 -5% 이하 또는 VIX 5일 +60% 이상


def _px(sym: str) -> pd.Series:
    f = PARQUET / _fname(sym)
    s = pd.read_parquet(f)["close"].dropna() if f.exists() else pd.Series(dtype=float)
    return s[~s.index.duplicated(keep="last")]


def exprank(s: pd.Series, minn: int = 1260) -> pd.Series:
    """현재까지 관측 중 현재값보다 작은 비율(동률 0.5). 미래 미포함."""
    out = np.full(len(s), np.nan)
    seen: list = []
    for i, v in enumerate(s.to_numpy(float)):
        if np.isnan(v):
            continue
        bisect.insort(seen, v)
        if len(seen) >= minn:
            lo, hi = bisect.bisect_left(seen, v), bisect.bisect_right(seen, v)
            out[i] = (lo + 0.5 * (hi - lo)) / len(seen)
    return pd.Series(out, index=s.index)


def level_of(x: float) -> str:
    return [n for lo, n in LEVELS if x >= lo][-1]


def features() -> pd.DataFrame:
    P = _px("^GSPC")
    P = P[P.index >= START]
    dxy = _px("DX-Y.NYB").reindex(P.index).ffill()
    r = np.log(P).diff()
    F = pd.DataFrame(index=P.index)
    F["below_200dma"] = -np.log(P / P.rolling(200).mean())
    F["neg_mom_12m"] = -(P / P.shift(252) - 1)
    F["drawdown_1y"] = -(P / P.rolling(252).max() - 1)
    F["realized_vol_3m"] = r.rolling(63).std() * np.sqrt(252)
    F["dollar_up_6m"] = dxy / dxy.shift(126) - 1
    return F


def _fwd(P: pd.Series, h: int):
    a = P.to_numpy(float)
    win = pd.Series(a[::-1]).shift(1).rolling(h, min_periods=h).min().to_numpy()[::-1]   # t+1..t+h 최저
    return pd.Series(win / a - 1, index=P.index), P.shift(-h) / P - 1


def run():
    F = features()
    P = _px("^GSPC").reindex(F.index)
    R = pd.DataFrame({c: exprank(F[c]) for c in F.columns})
    S = R.mean(axis=1, skipna=True).where(R.notna().any(axis=1))
    vix = _px("^VIX").reindex(F.index).ffill()
    hist = pd.DataFrame({"score": S, "spx": P, "vix": vix}).dropna(subset=["score"])
    hist["level"] = hist.score.map(level_of)
    dd6, ret6 = _fwd(P, H)
    T = pd.DataFrame({"score": S, "dd": dd6, "ret": ret6}).dropna()
    T["bad"] = (T.dd <= BAD).astype(float)
    T["level"] = T.score.map(level_of)
    by = T.groupby("level").agg(days=("bad", "size"), p_bad=("bad", "mean"), ret6_mean=("ret", "mean"), dd_mean=("dd", "mean"))
    by = by.reindex([n for _, n in LEVELS]).reset_index()

    # 충격 표지: 최근 발동 여부와 과거 발동 뒤 6개월
    ret5 = P / P.shift(5) - 1
    vix5 = vix / vix.shift(5) - 1
    shock = ((ret5 <= SHOCK_RET5) | (vix5 >= SHOCK_VIX5)).fillna(False)
    ev = shock[shock & ~shock.shift(1, fill_value=False)]
    ev = ev[~ev.index.to_series().diff().dt.days.lt(21).fillna(False)]   # 21일 안 재발동은 한 사건
    sh = pd.DataFrame({"dd": dd6.reindex(ev.index), "ret": ret6.reindex(ev.index)}).dropna()
    sh_stats = {"events": int(len(sh)), "p_bad": float(np.mean(sh.dd <= BAD)) if len(sh) else None,
                "ret6_median": float(sh.ret.median()) if len(sh) else None}

    # 코스피: 1년 고점 대비 낙폭 구간별 과거 기록
    K = _px("^KS11")
    K = K[K.index >= "1997-06-01"]
    kdd = K / K.rolling(252).max() - 1
    kd6, kr6 = _fwd(K, H)
    _, kr12 = _fwd(K, 252)
    KD = pd.DataFrame({"ddn": kdd, "dd6": kd6, "ret6": kr6, "ret12": kr12}).dropna()
    edges = [(-0.10, 0.0), (-0.20, -0.10), (-0.30, -0.20), (-0.45, -0.30), (-1.0, -0.45)]
    names = ["0~-10%", "-10~-20%", "-20~-30%", "-30~-45%", "-45% 이하"]
    krows = []
    for (lo, hi), nm in zip(edges, names):
        m = KD[(KD.ddn > lo) & (KD.ddn <= hi)]
        entry = m.index[m.index.to_series().diff().dt.days.fillna(999) > 120]   # 120일 넘게 비었다 들어온 횟수 = 독립 에피소드 근사
        krows.append({"bucket": nm, "days": int(len(m)), "episodes": int(len(entry)), "ret6_median": float(m.ret6.median()) if len(m) else np.nan,
                      "ret12_median": float(m.ret12.median()) if len(m) else np.nan, "p_more15": float((m.dd6 <= -0.15).mean()) if len(m) else np.nan,
                      "p_loss12": float((m.ret12 < 0).mean()) if len(m) else np.nan})
    kc = float(kdd.iloc[-1]) if len(kdd.dropna()) else np.nan
    kb = next((n for (lo, hi), n in zip(edges, names) if lo < kc <= hi), None) if kc == kc else None

    last = hist.index[-1]
    now = {"as_of": str(last.date()), "score": float(hist.score.iloc[-1]), "level": hist.level.iloc[-1],
           "parts": {PARTS[c]: (None if np.isnan(R[c].iloc[-1]) else float(R[c].iloc[-1])) for c in PARTS},
           "spx": float(P.dropna().iloc[-1]), "spx_dd": float(P.iloc[-1] / P.tail(252).max() - 1), "vix": float(vix.dropna().iloc[-1]),
           "ret5": float(ret5.iloc[-1]), "vix5": float(vix5.iloc[-1]),
           "shock": bool(shock.iloc[-1] or shock.tail(5).any()), "shock_stats": sh_stats,
           "kospi_dd": None if kc != kc else kc, "kospi_bucket": kb, "kospi_level": float(K.iloc[-1]) if len(K) else None,
           "kospi_prev_close": float(K.iloc[-2]) if len(K) > 1 else None,
           "prevalence": float(T.bad.mean())}
    return hist, by, pd.DataFrame(krows), now
