"""미국 신호 → 한국 전달 이벤트 스터디.
이벤트 = regime_backtest.conditions의 한국 관련 조건. 결과 = 이벤트 이후(한국 t+1부터) 외국인 순매수·한국 수익률.
foreign 결과는 '신호 다음 한국 거래일부터 h일 누적 순매수의 z'(승률 = 순매수 확률).
수익률 결과는 한국 t+1 종가 진입(entry lag 1)."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.features.common import Ctx

from .forward_returns import Fwd
from .regime_backtest import conditions, load_features
from .stats import SPLITS, cond_stats, split_valid

EVENTS = ["named:kr_riskon_5d", "named:kr_riskon_20d", "named:kr_riskoff_5d", "named:kr_riskoff_20d",
          "named:hawkish_squeeze", "named:easing_tailwind"]
FLOW_H = (1, 5, 20)
RET_H = (1, 3, 5, 10, 20)


def outcomes(ctx: Ctx, fw: Fwd) -> dict:
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(ctx.cal)
    o = {}
    for mkt in ("KOSPI", "KOSDAQ"):
        for h in FLOW_H:
            z = K[f"kr__{mkt}_foreign__net_{h}d_z"].shift(-h)  # t+1..t+h 누적
            o[(f"{mkt}_foreign_z", h)] = (z.to_numpy(float), None)
        for h in FLOW_H:
            z = K[f"kr__{mkt}_inst__net_{h}d_z"].shift(-h)
            o[(f"{mkt}_inst_z", h)] = (z.to_numpy(float), None)
    for h in RET_H:
        for name, sym in [("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11")]:
            o[(f"{name}_ret", h)] = fw.abs(sym, h)
        for name, sym in [("KR_Semis", "091160.KS"), ("Samsung", "005930.KS"), ("SKhynix", "000660.KS"),
                          ("KR_Auto", "091180.KS"), ("KR_Banks", "091170.KS")]:
            o[(f"{name}_rel", h)] = fw.rel(sym, "^KS11", h)
        o[("USDKRW_chg", h)] = fw.abs("KRW=X", h)
    return {k: v for k, v in o.items() if v is not None}


def run() -> pd.DataFrame:
    ctx = Ctx()
    fw = Fwd(ctx)
    F = load_features().reindex(ctx.cal)
    conds = conditions(F, ctx)
    out = outcomes(ctx, fw)
    valid = {(sp, h): split_valid(ctx.cal, sp, h + 1) for sp in SPLITS for h in set(FLOW_H) | set(RET_H)}
    rows = []
    for ev in EVENTS:
        mask = conds[ev][1]
        for (name, h), (y, mdd) in out.items():
            for sp in SPLITS:
                st = cond_stats(y, valid[(sp, h)], mask, h, None, min_obs=8)
                if st:
                    rows.append({"event": ev, "outcome": name, "h": h, "split": sp, **st})
    return pd.DataFrame(rows)
