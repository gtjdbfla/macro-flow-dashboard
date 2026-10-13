"""Lead/Lag: corr(x_t, y_{t+L}), L = 0/1/3/5/10/20. 일별 변화율(비겹침) 기준, 분할별 안정성 확인.
해석 주의: 통계적 선행 관계일 뿐 인과가 아니다. 한국 시계열 y의 t일은 미국 t일 개장 전에 끝나므로
'미국 → 한국 다음날'은 L=1, L=0은 한국이 미국 직전에 움직인 부분을 포함한다."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.features.common import Ctx

from .stats import SPLITS

LAGS = (0, 1, 3, 5, 10, 20)


def series(ctx: Ctx) -> dict:
    ret = lambda s: ctx.close(s).pct_change(fill_method=None)  # noqa: E731
    rs = lambda a, b: (np.log(ctx.close(a)) - np.log(ctx.close(b))).diff()  # noqa: E731
    M = pd.read_parquet(PARQUET / "features_macro.parquet").reindex(ctx.cal)
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(ctx.cal)
    s = {
        "d_UST10Y": M["macro__UST10Y__chg_1d"], "d_Real10Y": M["macro__Real10Y__chg_1d"],
        "DXY": ret("DX-Y.NYB"), "d_USDKRW": ret("KRW=X"),
        "QQQ": ret("QQQ"), "SOXX": ret("SOXX"), "QQQ-SPY": rs("QQQ", "SPY"), "SOXX-SPY": rs("SOXX", "SPY"),
        "SOXX-QQQ": rs("SOXX", "QQQ"),
        "KR_Semis": ret("091160.KS"), "KR_Semis-KOSPI": rs("091160.KS", "^KS11"), "KOSPI": ret("^KS11"),
        "Samsung": ret("005930.KS"), "SKhynix": ret("000660.KS"),
        "KOSPI_foreign_z": K["kr__KOSPI_foreign__net_1d_z"], "KOSDAQ_foreign_z": K["kr__KOSDAQ_foreign__net_1d_z"],
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in s.items()}


PAIRS = [
    ("d_UST10Y", "QQQ-SPY"), ("d_UST10Y", "SOXX-SPY"), ("d_Real10Y", "SOXX-SPY"), ("DXY", "QQQ"),
    ("DXY", "KOSPI_foreign_z"), ("d_USDKRW", "KOSPI_foreign_z"),
    ("SOXX", "KR_Semis"), ("SOXX-QQQ", "KR_Semis-KOSPI"), ("SOXX", "Samsung"), ("SOXX", "SKhynix"),
    ("SOXX", "KOSPI_foreign_z"), ("QQQ", "KOSPI_foreign_z"), ("QQQ", "KOSDAQ_foreign_z"), ("QQQ", "KOSPI"),
    ("KR_Semis", "SOXX"),  # 역방향 대조군: 한국이 미국을 선행하는가
]


def run() -> pd.DataFrame:
    ctx = Ctx()
    S = series(ctx)
    idx = ctx.cal
    rows = []
    for x, y in PAIRS:
        for L in LAGS:
            yy = S[y].shift(-L)
            for sp, (a, b) in SPLITS.items():
                m = (idx >= a) & (idx <= b)
                d = pd.concat([S[x][m], yy[m]], axis=1).dropna()
                n = len(d)
                if n < 100:
                    continue
                r = d.iloc[:, 0].corr(d.iloc[:, 1])
                rows.append({"x": x, "y": y, "lag": L, "split": sp, "n": n, "r": r,
                             "t": r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))})
    return pd.DataFrame(rows)
