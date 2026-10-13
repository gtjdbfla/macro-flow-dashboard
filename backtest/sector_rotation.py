"""섹터 회전 신호의 예측력: 섹터 feature가 이후 SPY 대비 초과수익을 예측하는가.
측정: 매일 11개 섹터 단면 순위상관(Spearman IC), 상위3−하위3 초과수익 스프레드.
겹침 제거: h행 간격으로 표본 추출(포지션 % h == 0). 분할 경계를 넘는 윈도우는 purge."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.features.common import Ctx
from src.features.sector_features import SECTORS

from .forward_returns import Fwd
from .stats import SPLITS, split_valid

FEATS = ["rs_5d", "rs_20d", "rs_60d", "rs_accel", "mom_12_1", "volume_z"]
REGIMES = ["ALL", "Goldilocks", "Reflation", "Slowdown", "Stagflation"]


def run(horizons=(5, 20, 60)) -> pd.DataFrame:
    ctx = Ctx()
    fw = Fwd(ctx)
    idx = ctx.cal
    S = pd.read_parquet(PARQUET / "features_sector.parquet").reindex(idx)
    regime = pd.read_parquet(PARQUET / "features_state.parquet")["regime"].reindex(idx)
    rows = []
    for h in horizons:
        Y = pd.DataFrame({s: fw.rel(s, "SPY", h)[0] for s in SECTORS if fw.close(s) is not None}, index=idx)
        for f in FEATS:
            X = pd.DataFrame({s: S[f"sector__{lab}__{f}"] for s, lab in SECTORS.items() if s in Y}, index=idx)
            common = X.notna() & Y.notna()
            n = common.sum(axis=1)
            Xr, Yr = X.where(common).rank(axis=1), Y.where(common).rank(axis=1)
            ic = Xr.corrwith(Yr, axis=1).where(n >= 8)
            top = Xr.gt((n - 3).to_numpy()[:, None])
            bot = Xr.le(3)
            spread = (Y.where(top & common).mean(axis=1) - Y.where(bot & common).mean(axis=1)).where(n >= 8)
            for sp in ("train", "val", "test"):
                v = split_valid(idx, sp, h + 1)
                samp = v & (np.arange(len(idx)) % h == 0)
                for rg in REGIMES:
                    m = samp if rg == "ALL" else samp & (regime == rg).to_numpy()
                    a, b = ic[m].dropna(), spread[m].dropna()
                    if len(a) < 12:
                        continue
                    rows.append({"feature": f, "h": h, "split": sp, "regime": rg, "n": len(a),
                                 "ic_mean": a.mean(), "ic_t": a.mean() / (a.std(ddof=1) / np.sqrt(len(a))),
                                 "ic_hit": float((a > 0).mean()), "spread_mean": b.mean(),
                                 "spread_t": b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))})
    return pd.DataFrame(rows)
