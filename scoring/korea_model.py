"""Korea Transmission Score — 백테스트에서 가장 안정적이었던 관계(미국 마감 → 다음 한국 거래일 외국인 순매수)를
선형 회귀 한 개로 요약한다. 설명 가능성 우선(계수 공개). 적합은 train+val(~2022), test는 표본 외 평가만.

y_t = 한국 t+1일 KOSPI 외국인 순매수 z,  X_t = [QQQ, SOXX, DXY 일수익률, 실질금리 1일 변화] (미국 t 마감) + 직전 한국 거래일 외국인 순매수 z.

v2(2026-10-14): 직전 외국인 z를 입력에 추가. 후보 4세트를 사전에 정하고 train(~2018) 적합→val(2019–22)로 선택→test(2023~)를 한 번만 봤다
(backtest/evidence_upgrade.korea_model_v2). 표본외 상관 0.33→0.46, R² 0.10→0.20. 외국인 수급이 하루 단위로 이어지는 성질을 반영한 것이다."""
import numpy as np
import pandas as pd

from backtest.stats import SPLITS
from src.data import PARQUET
from src.features.common import Ctx, exp_rank

XCOLS = ["QQQ", "SOXX", "DXY", "d_Real10Y", "lag_foreign_z"]
VERSION = "v2"


def build(ctx: Ctx):
    ret = lambda s: ctx.close(s).pct_change(fill_method=None)  # noqa: E731
    M = pd.read_parquet(PARQUET / "features_macro.parquet").reindex(ctx.cal)
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(ctx.cal)
    X = pd.DataFrame({"QQQ": ret("QQQ"), "SOXX": ret("SOXX"), "DXY": ret("DX-Y.NYB"),
                      "d_Real10Y": M["macro__Real10Y__chg_1d"],
                      "lag_foreign_z": K["kr__KOSPI_foreign__net_1d_z"].fillna(0.0)})  # t일까지의 한국 장 수급(미국 t 마감 전에 이미 확정)
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
    out = {"version": VERSION, "coef": dict(zip(["const"] + XCOLS, map(float, beta))), "fit_n": int(len(d)), "eval": {}}
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
