"""Feature 공통 도구.

규칙(look-ahead 방지):
  * 행 t의 값 = "미국 장 마감(t) 시점에 알 수 있었던 정보"로만 계산한다.
  * 발표 시차가 있는 데이터는 available_at 기준으로 달력에 올린다(`place`).
  * percentile/z-score는 expanding(과거만) 윈도우. 미래 표본을 쓰지 않는다.
  * Ctx(asof=T)는 T 이후 정보를 아예 보이지 않게 한다 -> tests/test_no_lookahead.py가 이를 검증.
"""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.data.yahoo import _fname

START = pd.Timestamp("2005-01-03")
MINP_D = 252  # 일별 percentile/z 최소 표본
MINP_M = 60   # 월별


def exp_rank(s: pd.Series, minp: int = MINP_D) -> pd.Series:
    """과거 값들 중 현재값보다 작은 비율(동률은 0.5). 현재·미래 미포함."""
    v = s.dropna()
    a = v.to_numpy(float)
    out = np.full(len(a), np.nan)
    for i in range(minp, len(a)):
        past = a[:i]
        out[i] = ((past < a[i]).sum() + 0.5 * (past == a[i]).sum()) / i
    return pd.Series(out, index=v.index).reindex(s.index)


def exp_z(s: pd.Series, minp: int = MINP_D) -> pd.Series:
    sd = s.expanding(minp).std()
    return (s - s.expanding(minp).mean()) / sd.where(sd > 0)


def slope(s: pd.Series, w: int = 60) -> pd.Series:
    """w구간 회귀 기울기 x w = 구간 내 적합 총변화량."""
    x = np.arange(w) - (w - 1) / 2
    d = (x ** 2).sum()
    return s.rolling(w).apply(lambda v: (x * v).sum() / d, raw=True) * w


def pct_safe(s: pd.Series, h: int) -> pd.Series:
    """음수 가격(2020 WTI 등)이 끼면 NaN."""
    prev = s.shift(h)
    return ((s / prev) - 1).where((s > 0) & (prev > 0))


def place(F: pd.DataFrame, avail: pd.Series, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """관측일 인덱스의 feature를 available_at 기준으로 달력에 as-of 배치."""
    G = F.copy()
    G.index = pd.DatetimeIndex(avail.reindex(F.index).values)
    G = G[~G.index.isna()]
    G = G[~G.index.duplicated(keep="last")].sort_index()
    return G.reindex(G.index.union(cal)).ffill().reindex(cal)


class Ctx:
    """원천 데이터 접근. asof가 있으면 그 이후 정보는 보이지 않는다."""

    def __init__(self, asof=None):
        self.asof = pd.Timestamp(asof) if asof is not None else None
        spy = self._raw_px("SPY")
        self.cal = spy.index[spy.index >= START]
        self._cache = {}

    # ---- 가격 ----
    def _raw_px(self, sym):
        f = PARQUET / _fname(sym)
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        return df[df.index <= self.asof] if self.asof is not None else df

    def px(self, sym):
        if sym not in self._cache:
            df = self._raw_px(sym)
            self._cache[sym] = None if df is None or df.empty else df
        return self._cache[sym]

    def close(self, sym, adj=True):
        df = self.px(sym)
        if df is None:
            return None
        s = df["adjclose" if adj else "close"]
        return s.reindex(s.index.union(self.cal)).ffill().reindex(self.cal)

    def volume(self, sym):
        df = self.px(sym)
        if df is None:
            return None
        return df["volume"].reindex(self.cal)

    # ---- FRED ----
    def fred_daily_native(self, sid, lag_days=1):
        """(관측일 인덱스 Series, available_at Series)."""
        f = PARQUET / f"fred_{sid}.parquet"
        if not f.exists():
            return None, None
        s = pd.read_parquet(f)["value"].dropna()
        avail = pd.Series(s.index + pd.Timedelta(days=lag_days), index=s.index)
        if self.asof is not None:
            keep = avail <= self.asof
            s, avail = s[keep], avail[keep]
        return (s, avail) if len(s) else (None, None)

    def fred_on_cal(self, sid, lag_days=1):
        s, avail = self.fred_daily_native(sid, lag_days)
        if s is None:
            return None
        return place(s.to_frame("v"), avail, self.cal)["v"]

    def first_print(self, sid, default_lag, max_lag):
        """vintage 데이터에서 관측일별 최초 공표값. 최초 공표가 비정상적으로 늦으면
        (DB 적재 이전 과거분) date+default_lag로 근사하고 approx=True로 표시."""
        f = PARQUET / f"fredv_{sid}.parquet"
        if not f.exists():
            return None
        v = pd.read_parquet(f)
        if self.asof is not None:
            v = v[v["available_at"] <= self.asof]
        if v.empty:
            return None
        fp = v.sort_values("available_at").groupby("date", as_index=False).first()
        late = (fp["available_at"] - fp["date"]).dt.days > max_lag
        fp["approx"] = late
        fp.loc[late, "available_at"] = fp.loc[late, "date"] + pd.Timedelta(days=default_lag)
        if self.asof is not None:
            fp = fp[fp["available_at"] <= self.asof]
        return fp.set_index("date").sort_index()


def feats_series(s: pd.Series, name: str, kind="diff", hs=(1, 5, 20, 60), unit="d",
                 minp=MINP_D, with_rank=True, key_h=None) -> pd.DataFrame:
    """일/주 단위 시계열의 표준 feature 묶음. 인덱스는 s의 인덱스(관측 순서)."""
    out = {f"{name}__level": s}
    for h in hs:
        out[f"{name}__chg_{h}{unit}"] = s.diff(h) if kind == "diff" else pct_safe(s, h)
    big = key_h or max(hs)  # slope/z/rank 기준 구간
    ch = out[f"{name}__chg_{big}{unit}"]
    out[f"{name}__slope_{big}{unit}"] = slope(s, big)
    out[f"{name}__chg_{big}{unit}_z"] = exp_z(ch, minp)
    if with_rank:
        out[f"{name}__level_rank"] = exp_rank(s, minp)
        out[f"{name}__chg_{big}{unit}_rank"] = exp_rank(ch, minp)
    return pd.DataFrame(out)
