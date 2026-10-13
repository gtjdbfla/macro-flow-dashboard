"""조건부 forward 성과 통계.

겹치는 관측치 문제: 조건 충족일 t의 h일 수익률은 t+1..t+h-1의 것과 겹쳐 '독립 표본 수'가 N보다 훨씬 적다.
-> 조건 충족일을 'episode'(간격 > h 로 끊김)로 묶고, episode 단위로 군집-강건 표준오차를 낸다.
   n_eff = episode 수. CI는 t분포(df = n_eff-1). 기준선(무조건 평균)의 불확실성은 N이 커서 무시한다.
"""
import numpy as np
from scipy import stats as sps

SPLITS = {"train": ("2005-01-01", "2018-12-31"), "val": ("2019-01-01", "2022-12-31"),
          "test": ("2023-01-01", "2100-01-01"), "all": ("2005-01-01", "2100-01-01")}
_T = {df: sps.t.ppf(0.975, df) for df in range(1, 400)}


def tcrit(df: int) -> float:
    return _T.get(df, 1.96)


def episodes(pos: np.ndarray, gap: int) -> np.ndarray:
    return np.cumsum(np.r_[0, np.diff(pos) > gap]) if len(pos) else np.array([], int)


def split_valid(index, split: str, horizon_rows: int) -> np.ndarray:
    """분할 구간 안에서, 미래 윈도우(horizon_rows)가 구간 끝을 넘지 않는 행만 True (purge)."""
    a, b = (np.datetime64(x) for x in SPLITS[split])
    d = index.values
    inside = (d >= a) & (d <= b)
    pos = np.flatnonzero(inside)
    ok = np.zeros(len(d), bool)
    if len(pos):
        ok[pos[0]: max(pos[0], pos[-1] - horizon_rows + 1)] = True
    return ok & inside


def cond_stats(y: np.ndarray, valid: np.ndarray, mask: np.ndarray, h: int, mdd=None, min_obs: int = 5):
    ok = valid & np.isfinite(y)
    if ok.sum() < 50:
        return None
    sel = ok & mask
    r = y[sel]
    if len(r) < min_obs:
        return None
    base = y[ok]
    cid = episodes(np.flatnonzero(sel), h)
    k = int(cid[-1]) + 1
    C = np.bincount(cid).astype(float)
    S = np.bincount(cid, weights=r)
    m = r.mean()
    exc = m - base.mean()
    se = ci = np.nan
    if k >= 3:
        u = S - m * C
        se = np.sqrt(k / (k - 1) * (u ** 2).sum()) / C.sum()
        ci = tcrit(k - 1) * se
    row = {"n": len(r), "n_eff": k, "mean": m, "median": float(np.median(r)), "std": float(r.std(ddof=1)),
           "win": float((r > 0).mean()), "base_mean": float(base.mean()), "base_win": float((base > 0).mean()),
           "excess": exc, "se": se, "ci_lo": exc - ci, "ci_hi": exc + ci}
    if mdd is not None:
        mm = mdd[sel]
        row["mdd_mean"] = float(np.nanmean(mm)) if np.isfinite(mm).any() else np.nan
        row["mdd_worst"] = float(np.nanmin(mm)) if np.isfinite(mm).any() else np.nan
    return row
