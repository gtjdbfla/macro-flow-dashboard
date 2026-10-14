"""Big4 CAPEX 사이클 이벤트 스터디. 질문: 하이퍼스케일러 투자 지표가 공개된 뒤 반도체/AI 주식은 어떻게 움직였나.

이벤트 = Big4(MSFT·GOOGL·AMZN·META) 분기 실적이 모두 공시된 날(마지막 공시일). 그날 이후 5/20/60 거래일 수익률을
'이벤트 조건별'로 무조건 기준선과 비교한다. 표본: 2014-09~ 약 45개 분기(AI 시대 12분기만이 아니라 이전 CAPEX 사이클 포함).
통계는 backtest.stats.cond_stats(독립 사례 = 이벤트 수, 군집-강건 CI)."""
import numpy as np
import pandas as pd

from src.ai import metrics as M
from src.features.common import Ctx

from .forward_returns import Fwd
from .stats import cond_stats, split_valid

ASSETS = ["SOXX", "SMH", "NVDA", "000660.KS", "005930.KS", "QQQ", "SPY"]
HS = (5, 20, 60)


def events(ctx: Ctx) -> pd.DataFrame:
    b, avail = M.all_companies()["BIG4"]
    o = b.dropna(subset=["capex_yoy"]).copy()
    o["avail"] = avail.reindex(o.index)
    o["capex_accel"] = o["capex_yoy"] - o["capex_yoy"].shift(1)
    o["rev_accel"] = o["rev_yoy"] - o["rev_yoy"].shift(1)
    pos = ctx.cal.searchsorted(o["avail"].values.astype("datetime64[ns]"))  # 공시 당일 또는 그 이후 첫 거래일
    o["row"] = pos
    return o[o["row"] < len(ctx.cal)].dropna(subset=["capex_accel"])


def run() -> pd.DataFrame:
    ctx = Ctx()
    fw = Fwd(ctx)
    ev = events(ctx)
    n = len(ctx.cal)
    conds = {
        "capex_accelerating": ev["capex_accel"] > 0, "capex_decelerating": ev["capex_accel"] < 0,
        "fcf_falling": ev["fcf_chg"] < 0, "fcf_rising": ev["fcf_chg"] >= 0,
        "capex_outgrowing_revenue_by_30pp": (ev["capex_yoy"] - ev["rev_yoy"]) > 0.30,
        "revenue_accelerating": ev["rev_accel"] > 0,
    }
    rows = []
    for cname, cond in conds.items():
        mask = np.zeros(n, bool)
        mask[ev.loc[cond, "row"].values] = True
        for asset in ASSETS:
            for h in HS:
                res = fw.abs(asset, h)
                if res is None:
                    continue
                for sp in ("all", "train", "val", "test"):
                    st = cond_stats(res[0], split_valid(ctx.cal, sp, h + 1), mask, h, res[1], min_obs=4)
                    if st:
                        rows.append({"cond": cname, "asset": asset, "h": h, "split": sp, **st})
    return pd.DataFrame(rows)
