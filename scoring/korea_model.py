"""Korea Transmission Score — 백테스트에서 가장 안정적이었던 관계(미국 마감 → 다음 한국 거래일 외국인 순매수)를
선형 회귀 한 개로 요약한다. 설명 가능성 우선(계수 공개). 적합은 train+val(~2022), test는 표본 외 평가만.

y_t = 한국 t+1일 KOSPI 외국인 순매수 z,  X_t = [QQQ, SOXX, DXY 일수익률, 실질금리 1일 변화] (미국 t 마감)."""
import numpy as np
import pandas as pd

from backtest.stats import SPLITS
from src.data import PARQUET
from src.features.common import Ctx, exp_rank

XCOLS = ["QQQ", "SOXX", "DXY", "d_Real10Y"]


def build(ctx: Ctx):
    ret = lambda s: ctx.close(s).pct_change(fill_method=None)  # noqa: E731
    M = pd.read_parquet(PARQUET / "features_macro.parquet").reindex(ctx.cal)
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(ctx.cal)
    X = pd.DataFrame({"QQQ": ret("QQQ"), "SOXX": ret("SOXX"), "DXY": ret("DX-Y.NYB"),
                      "d_Real10Y": M["macro__Real10Y__chg_1d"]})
    y = K["kr__KOSPI_foreign__net_1d_z"].shift(-1)  # 다음 행 = 다음 미국 영업일 사이의 한국 순매수
    return X, y


def run():
    ctx = Ctx()
    X, y = build(ctx)
    idx = ctx.cal
    fit_m = (idx >= "2005-01-01") & (idx <= "2022-12-31")
    d = pd.concat([X, y.rename("y")], axis=1)[fit_m].dropna()
    mu, sd = d[XCOLS].mean(), d[XCOLS].std()
    Z = (d[XCOLS] - mu) / sd
    A = np.c_[np.ones(len(Z)), Z.values]
    beta, *_ = np.linalg.lstsq(A, d["y"].values, rcond=None)
    Zall = (X[XCOLS] - mu) / sd
    pred = pd.Series(np.c_[np.ones(len(Zall)), Zall.fillna(0).values] @ beta, index=idx).where(Zall.notna().all(axis=1))
    out = {"coef": dict(zip(["const"] + XCOLS, map(float, beta))), "fit_n": int(len(d)), "eval": {}}
    for k in ("train", "val", "test"):
        a, b = SPLITS[k]
        m = (idx >= a) & (idx <= b)
        e = pd.concat([pred[m], y[m]], axis=1).dropna()
        r = e.iloc[:, 0].corr(e.iloc[:, 1])
        ss = ((e.iloc[:, 1] - e.iloc[:, 0]) ** 2).sum() / ((e.iloc[:, 1] - e.iloc[:, 1].mean()) ** 2).sum()
        out["eval"][k] = {"corr": float(r), "r2_vs_mean": float(1 - ss), "n": int(len(e))}
    score = 100 * exp_rank(pred, 252)
    hist = pd.DataFrame({"pred_foreign_z": pred, "actual_next_foreign_z": y, "korea_score": score})
    return hist, out
