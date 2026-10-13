"""Layer 3 — Risk Appetite. 점수화는 Phase 4(백테스트 후). 여기서는 구성 요소만."""
import numpy as np
import pandas as pd

from .common import Ctx, exp_rank, feats_series, place


def build(ctx: Ctx) -> pd.DataFrame:
    parts = []
    for sym, name in [("^VIX", "VIX"), ("^MOVE", "MOVE")]:
        c = ctx.close(sym, adj=False)
        if c is not None:
            parts.append(feats_series(c.dropna(), name, "diff", hs=(1, 5, 20)).reindex(ctx.cal))
    hyg, lqd = ctx.close("HYG"), ctx.close("LQD")
    if hyg is not None and lqd is not None:
        r = np.log(hyg / lqd).dropna()  # 하이일드/투자등급 상대가격 -> 신용 위험선호
        parts.append(feats_series(r, "HYG_LQD", "diff", hs=(5, 20, 60)).drop(columns="HYG_LQD__level")
                     .reindex(ctx.cal))
    # FRED 스프레드는 라이선스로 2023-10~만 존재(커버리지 참고용)
    for sid, name in [("BAMLH0A0HYM2", "HY_OAS"), ("BAMLC0A0CM", "IG_OAS")]:
        s, avail = ctx.fred_daily_native(sid, 1)
        if s is not None and len(s) > 300:
            parts.append(place(feats_series(s, name, "diff", hs=(5, 20, 60), minp=120), avail, ctx.cal))
    spy = ctx.close("SPY")
    dd = spy / spy.cummax() - 1
    parts.append(pd.DataFrame({
        "SPY__drawdown": dd,
        "SPY__rv_20d": np.log(spy).diff().rolling(20).std() * np.sqrt(252),
        "SPY__drawdown_rank": exp_rank(dd),
    }))
    return pd.concat(parts, axis=1).add_prefix("risk__")
