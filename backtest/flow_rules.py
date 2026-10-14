"""'자금이 많이 들어온/나간 섹터' 규칙의 과거 성과(홈 화면의 '다음 자금 이동 후보' 근거).

규칙 US: 최근 20일 ETF 순유입(%순자산) 상위 3 섹터 vs 하위 3 섹터의 이후 20일 SPY 대비 초과수익 차이(상위−하위).
         음수면 '많이 들어온 섹터가 이후 덜 올랐다'(역방향).
규칙 KR: 업종 외국인 순매수 비중(60일) 상위 3 vs 하위 3 업종의 이후 20일 KOSPI 대비 초과수익 차이(상위−하위).
         양수면 '외국인이 많이 산 업종이 이후에도 더 올랐다'(지속).
표본: 20거래일 간격(비겹침), 분할 경계 purge. 결과는 train/val/test/all로 나눠 보고, 구간별 방향이 같은지 본다."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.features.common import Ctx
from src.features.sector_features import SECTORS

from .forward_returns import Fwd
from .stats import split_valid

KR_ETF = {"Semis": "091160.KS", "Auto": "091180.KS", "Banks": "091170.KS", "Healthcare": "143860.KS",
          "EnergyChem": "117460.KS", "Battery": "305720.KS", "Shipbuilding": "466920.KS", "Defense": "463250.KS"}
H = 20


def _spreads(X: pd.DataFrame, Y: pd.DataFrame, idx, min_n: int):
    common = X.notna() & Y.notna()
    n = common.sum(axis=1)
    Xr = X.where(common)
    ranks = Xr.rank(axis=1, ascending=False, method="first")
    top = ranks <= 3
    bot = ranks > (n.to_numpy()[:, None] - 3)
    top_m = Y.where(top & common).mean(axis=1)
    bot_m = Y.where(bot & common).mean(axis=1)
    return pd.DataFrame({"top": top_m, "bottom": bot_m, "spread": top_m - bot_m}).where(n >= min_n)


def _summ(d: pd.DataFrame, rule: str, split: str, sign: int) -> dict:
    s = d["spread"].dropna()
    n = len(s)
    if n < 6:
        return {}
    return {"rule": rule, "split": split, "n": n, "mean_spread": s.mean() * 100,
            "t": s.mean() / (s.std(ddof=1) / np.sqrt(n)), "hit": float(((s * sign) > 0).mean()) * 100,
            "top_mean": d["top"].dropna().mean() * 100, "bottom_mean": d["bottom"].dropna().mean() * 100}


def run() -> pd.DataFrame:
    ctx = Ctx()
    fw = Fwd(ctx)
    idx = ctx.cal
    S = pd.read_parquet(PARQUET / "features_sector.parquet").reindex(idx)
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(idx)
    rows = []
    sets = {
        "us_flow": (pd.DataFrame({lab: S[f"sector__{lab}__flow_20d"] for lab in SECTORS.values() if f"sector__{lab}__flow_20d" in S}),
                    pd.DataFrame({lab: fw.rel(sym, "SPY", H)[0] for sym, lab in SECTORS.items() if fw.close(sym) is not None}, index=idx), 8, -1),
        "kr_foreign": (pd.DataFrame({k: K[f"kr__{k}_foreign__ratio_60d"] for k in KR_ETF if f"kr__{k}_foreign__ratio_60d" in K}),
                       pd.DataFrame({k: fw.rel(e, "^KS11", H)[0] for k, e in KR_ETF.items() if fw.close(e) is not None}, index=idx), 6, +1),
    }
    for rule, (X, Y, min_n, sign) in sets.items():
        Y = Y[[c for c in X.columns if c in Y.columns]]
        X = X[Y.columns]
        d = _spreads(X, Y, idx, min_n)
        samp = np.arange(len(idx)) % H == 0
        for sp in ("train", "val", "test", "all"):
            m = split_valid(idx, sp, H + 1) & samp
            r = _summ(d[m], rule, sp, sign)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)
