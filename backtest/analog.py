"""Historical Analog — 현재와 비슷한 과거 날짜 찾기.
상태 벡터(모두 expanding 순위 0~1이라 미래 정보 없음) 간 RMS 거리. 유사 = 전체 거리의 하위 10%.
겹치는 날들은 episode(간격 > 10영업일)로 묶어 '독립 사례 수'를 함께 보여준다.
이후 수익률은 5/20/60/120일 분포(평균·중앙값·10/90분위·승률)를 무조건 기준선과 비교한다."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.features.common import Ctx, exp_rank

from .forward_returns import Fwd, bench_for
from .regime_backtest import ASSETS, BENCHES
from .stats import episodes

VEC = [("성장", "state__growth_z", True), ("물가", "state__inflation_z", True),
       ("정책금리 6M", "state__policy_chg_6m", True),
       ("순유동성 60D", "liq__NetLiquidity__chg_60d_rank", False), ("달러 60D", "macro__DXY__chg_60d_rank", False),
       ("美10Y 60D", "macro__UST10Y__chg_60d_rank", False), ("실질금리 60D", "macro__Real10Y__chg_60d_rank", False),
       ("VIX 수준", "risk__VIX__level_rank", False)]
HS = (5, 20, 60, 120)
PURGE = 130  # 최근 130영업일은 후보에서 제외(미래 수익률 미확정·현재와 중복)


def state_matrix():
    parts = [pd.read_parquet(PARQUET / f"features_{n}.parquet") for n in ("macro", "liquidity", "risk")]
    parts.append(pd.read_parquet(PARQUET / "features_state.parquet").add_prefix("state__"))
    M = pd.concat(parts, axis=1)
    X = pd.DataFrame({lab: (exp_rank(M[c], 252) if need_rank else M[c]) for lab, c, need_rank in VEC})
    return X


def run():
    ctx = Ctx()
    fw = Fwd(ctx)
    X = state_matrix().reindex(ctx.cal)
    valid = X.dropna()
    cur = valid.iloc[-1]
    cand = valid.iloc[:-PURGE]
    d = np.sqrt(((cand - cur) ** 2).mean(axis=1))
    thr = d.quantile(0.10)
    sim = d[d <= thr]
    pos = np.flatnonzero(ctx.cal.isin(sim.index))
    ep = episodes(pos, 10)
    meta = {"as_of": str(valid.index[-1].date()), "n_candidates": int(len(cand)), "n_days": int(len(sim)),
            "n_episodes": int(ep[-1] + 1) if len(ep) else 0, "threshold": float(thr),
            "nearest": float(d.min()), "median_all": float(d.median()),
            "vector": {k: float(v) for k, v in cur.items()}}
    nb = pd.DataFrame({"date": sim.index, "dist": sim.values, "episode": ep})
    nb["is_nearest"] = nb.dist.rank() <= 10
    sim_mask = np.zeros(len(ctx.cal), bool)
    sim_mask[pos] = True
    rows = []
    for asset in ASSETS:
        for kind in (["abs"] + (["rel"] if asset not in BENCHES else [])):
            for h in HS:
                res = fw.abs(asset, h) if kind == "abs" else fw.rel(asset, bench_for(asset), h)
                if res is None:
                    continue
                y = res[0]
                ok = np.isfinite(y)
                base = y[ok & (np.arange(len(y)) < len(y) - PURGE)]
                r = y[sim_mask & ok]
                if len(r) < 10 or len(base) < 100:
                    continue
                rows.append({"asset": asset, "kind": kind, "h": h, "n": len(r),
                             "n_eff": int(episodes(np.flatnonzero(sim_mask & ok), h)[-1] + 1),
                             "mean": r.mean(), "median": np.median(r), "p10": np.quantile(r, .1),
                             "p90": np.quantile(r, .9), "win": (r > 0).mean(),
                             "base_median": np.median(base), "base_win": (base > 0).mean(),
                             "base_p10": np.quantile(base, .1), "base_p90": np.quantile(base, .9)})
    return meta, nb, pd.DataFrame(rows)
