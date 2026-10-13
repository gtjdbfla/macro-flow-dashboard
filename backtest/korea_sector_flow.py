"""한국 업종 자금흐름(외국인·기관 순매수/거래대금)이 이후 업종 ETF의 KOSPI 대비 초과수익을 예측하는가.
8개 업종 단면 Spearman IC(비겹침 표본, 분할 purge). 진입은 한국 t+1 종가(미국 마감 신호 기준 보수적)."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.data.universe import KR_SECTOR_STOCKS
from src.features.common import Ctx

from .forward_returns import Fwd
from .stats import split_valid

ETF = {"Semis": "091160.KS", "Auto": "091180.KS", "Banks": "091170.KS", "Healthcare": "143860.KS",
       "EnergyChem": "117460.KS", "Battery": "305720.KS", "Shipbuilding": "466920.KS", "Defense": "463250.KS"}
FEATS = [("foreign", "ratio_5d"), ("foreign", "ratio_20d"), ("foreign", "ratio_60d"),
         ("inst", "ratio_5d"), ("inst", "ratio_20d"), ("inst", "ratio_60d")]


def run(horizons=(5, 20, 60)) -> pd.DataFrame:
    ctx = Ctx()
    fw = Fwd(ctx)
    idx = ctx.cal
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(idx)
    rows = []
    for h in horizons:
        Y = pd.DataFrame({s: fw.rel(e, "^KS11", h)[0] for s, e in ETF.items() if fw.close(e) is not None}, index=idx)
        for who, f in FEATS:
            X = pd.DataFrame({s: K.get(f"kr__{s}_{who}__{f}") for s in Y.columns}, index=idx)
            common = X.notna() & Y.notna()
            n = common.sum(axis=1)
            ic = X.where(common).rank(axis=1).corrwith(Y.where(common).rank(axis=1), axis=1).where(n >= 5)
            for sp in ("train", "val", "test", "all"):
                m = split_valid(idx, sp, h + 1) & (np.arange(len(idx)) % h == 0)
                a = ic[m].dropna()
                if len(a) >= 12:
                    rows.append({"who": who, "feature": f, "h": h, "split": sp, "n": len(a), "ic_mean": a.mean(),
                                 "ic_t": a.mean() / (a.std(ddof=1) / np.sqrt(len(a))), "ic_hit": float((a > 0).mean())})
    return pd.DataFrame(rows)
