"""Regime/조건 백테스트. 조건은 코드에 미리 고정(사후 추가 금지) — 결과를 본 뒤 조건을 늘리거나 튜닝하지 않는다.
분할: train 2005-2018 / val 2019-2022 / test 2023~, 분할 경계를 넘는 forward 윈도우는 제거(purge)."""
import numpy as np
import pandas as pd

from src.features.common import Ctx
from src.features.sector_features import INDUSTRIES, SECTORS

from .forward_returns import Fwd, bench_for
from .stats import SPLITS, cond_stats, split_valid

HORIZONS = (5, 20, 60, 120)
US_BROAD = ["SPY", "QQQ", "IWM", "TLT", "GC=F", "CL=F"]
KR_ASSETS = ["^KS11", "^KQ11", "069500.KS", "091160.KS", "091180.KS", "091170.KS", "143860.KS",
             "117460.KS", "305720.KS", "466920.KS", "463250.KS", "005930.KS", "000660.KS"]
ASSETS = US_BROAD + list(SECTORS) + list(INDUSTRIES) + KR_ASSETS
BENCHES = {"SPY", "^KS11"}


def load_features():
    from src.data import PARQUET
    parts = [pd.read_parquet(PARQUET / f"features_{n}.parquet") for n in ("macro", "liquidity", "state")]
    return pd.concat(parts, axis=1)


def conditions(F: pd.DataFrame, ctx: Ctx) -> dict:
    """이름 -> (그룹, bool 마스크). 마스크는 해당 시점 정보로만 정의된다."""
    c = {}
    S = F
    for col, vals in [("regime", ["Goldilocks", "Reflation", "Slowdown", "Stagflation"]),
                      ("policy", ["Easing", "Neutral", "Tightening"]),
                      ("liquidity", ["Expanding", "Neutral", "Contracting"]),
                      ("dollar", ["Weak", "Neutral", "Strong"]),
                      ("growth", ["Accelerating", "Decelerating"]), ("inflation", ["Accelerating", "Decelerating"])]:
        for v in vals:
            c[f"{col}={v}"] = ("single", (S[col] == v).to_numpy())
    for k in S["state_key"].dropna().unique():
        c[f"state={k}"] = ("combo", (S["state_key"] == k).to_numpy())

    def rk(name):
        return S[name]
    hi = lambda n: (rk(n) > 2 / 3).to_numpy()  # noqa: E731
    lo = lambda n: (rk(n) < 1 / 3).to_numpy()  # noqa: E731
    r = lambda sym, h: ctx.close(sym).pct_change(h, fill_method=None)  # noqa: E731
    # 스펙 예시 1: 10Y↑ + 실질금리↑ + DXY↑ + 유동성↓  (변화율 60D의 과거 대비 상위 1/3)
    c["named:hawkish_squeeze"] = ("named", hi("macro__UST10Y__chg_60d_rank") & hi("macro__Real10Y__chg_60d_rank")
                                  & hi("macro__DXY__chg_60d_rank") & (S["liquidity"] == "Contracting").to_numpy())
    c["named:easing_tailwind"] = ("named", lo("macro__UST10Y__chg_60d_rank") & lo("macro__Real10Y__chg_60d_rank")
                                  & lo("macro__DXY__chg_60d_rank") & (S["liquidity"] == "Expanding").to_numpy())
    # 스펙 예시 2(한국): SOXX↑ + NASDAQ↑ + DXY↓ + USDKRW↓  /  실질금리↑ + DXY↑ + SOXX↓
    for h in (5, 20):
        c[f"named:kr_riskon_{h}d"] = ("named", ((r("SOXX", h) > 0) & (r("QQQ", h) > 0) & (r("DX-Y.NYB", h) < 0)
                                                & (r("KRW=X", h) < 0)).fillna(False).to_numpy())
        rr = F[f"macro__Real10Y__chg_{h}d"] if f"macro__Real10Y__chg_{h}d" in F else F["macro__Real10Y__chg_20d"]
        c[f"named:kr_riskoff_{h}d"] = ("named", ((rr > 0) & (r("DX-Y.NYB", h) > 0) & (r("SOXX", h) < 0))
                                       .fillna(False).to_numpy())
    return c


def run(horizons=HORIZONS) -> pd.DataFrame:
    ctx = Ctx()
    fw = Fwd(ctx)
    F = load_features().reindex(ctx.cal)
    conds = conditions(F, ctx)
    idx = ctx.cal
    valid = {(s, h): split_valid(idx, s, h + 1) for s in SPLITS for h in horizons}
    rows = []
    targets = [(a, "abs") for a in ASSETS] + [(a, "rel") for a in ASSETS if a not in BENCHES]
    for asset, kind in targets:
        for h in horizons:
            res = fw.abs(asset, h) if kind == "abs" else fw.rel(asset, bench_for(asset), h)
            if res is None:
                continue
            y, mdd = res
            for cname, (grp, mask) in conds.items():
                for sp in SPLITS:
                    st = cond_stats(y, valid[(sp, h)], mask, h, mdd)
                    if st:
                        rows.append({"cond": cname, "group": grp, "asset": asset, "kind": kind, "h": h,
                                     "split": sp, **st})
    return pd.DataFrame(rows)


def replication(df: pd.DataFrame, min_eff: int = 5) -> pd.DataFrame:
    """train에서 CI가 0을 벗어난 효과가 val·test에서도 같은 부호로 재현되는가."""
    key = ["cond", "group", "asset", "kind", "h"]
    w = df[df.split.isin(["train", "val", "test"])].pivot_table(
        index=key, columns="split", values=["excess", "ci_lo", "ci_hi", "n_eff", "win", "base_win"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    need = ["excess_train", "excess_val", "excess_test", "ci_lo_train", "ci_hi_train", "n_eff_train",
            "n_eff_val", "n_eff_test"]
    w = w.dropna(subset=need)
    enough = (w.n_eff_train >= min_eff) & (w.n_eff_val >= min_eff) & (w.n_eff_test >= min_eff)
    sig = (w.ci_lo_train > 0) | (w.ci_hi_train < 0)
    same = (np.sign(w.excess_train) == np.sign(w.excess_val)) & (np.sign(w.excess_train) == np.sign(w.excess_test))
    w["testable"] = enough
    w["train_sig"] = sig
    w["replicated"] = enough & sig & same
    return w
