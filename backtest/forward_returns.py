"""Forward return 계산.

진입 시점: 신호는 '미국 마감(t)' 기준. 미국 자산은 t 종가 진입, 한국 자산은 한국 t일 장이 이미 끝났으므로
t+1 종가 진입(entry lag 1)으로 계산한다 -> 한국 수익률에 신호 이전 정보가 섞이지 않는다.
수익률은 조정종가 기준 단순수익률, MDD는 진입 후 h일 안의 최저 낙폭."""
import numpy as np
import pandas as pd

from src.features.common import Ctx

KR_SUFFIX = (".KS",)
KR_INDEX = {"^KS11", "^KQ11"}


def entry_lag(sym: str) -> int:
    return 1 if (sym in KR_INDEX or sym.endswith(KR_SUFFIX)) else 0


class Fwd:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.index = ctx.cal
        self._c = {}
        self._f = {}

    def close(self, sym):
        if sym not in self._c:
            self._c[sym] = self.ctx.close(sym)
        return self._c[sym]

    def abs(self, sym, h):
        """(y, mdd) numpy 배열. 데이터 없으면 None."""
        key = (sym, h)
        if key in self._f:
            return self._f[key]
        c = self.close(sym)
        if c is None:
            self._f[key] = None
            return None
        cs = c.shift(-entry_lag(sym))
        y = cs.shift(-h) / cs - 1
        fmin = cs.iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].shift(-1)  # t+1..t+h 최저가
        mdd = np.minimum(fmin / cs - 1, 0)
        self._f[key] = (y.to_numpy(float), mdd.to_numpy(float))
        return self._f[key]

    def rel(self, sym, bench, h):
        a, b = self.abs(sym, h), self.abs(bench, h)
        if a is None or b is None:
            return None
        return a[0] - b[0], None


def bench_for(sym):
    return "^KS11" if entry_lag(sym) else "SPY"
