import numpy as np
import pandas as pd

from src.ai import cloud


def test_incomplete_cloud_coverage_does_not_become_a_three_company_aggregate():
    h = pd.DataFrame({
        "ticker": ["AMZN", "GOOGL", "MSFT"],
        "quarter_end": pd.to_datetime(["2026-09-30"] * 3),
        "cloud_rev": [46, 31, np.nan],
        "cloud_rev_prior": [33, 15, np.nan],
        "cloud_opinc": [18, 11, np.nan],
        "cloud_opinc_prior": [11, 4, np.nan],
        "yoy": [.39, 1.03, np.nan],
        "yoy_chg_pp": [3, 21, np.nan],
        "margin": [.39, .36, np.nan],
        "margin_chg_pp": [4, 13, np.nan],
        "azure_yoy": [np.nan, np.nan, .48],
    })
    assert cloud.aggregate(h).empty
    assert cloud.aggregate(h[h.ticker != "MSFT"]).empty
    full = h.copy()
    full.loc[2, ["cloud_rev", "cloud_rev_prior", "cloud_opinc", "cloud_opinc_prior"]] = [40, 30, 16, 12]
    assert cloud.aggregate(full).iloc[0].rev == 117
    f7 = cloud.flags(h, None)[0]
    assert f7["level"] == "긍정"
    assert "비교 가능한 2곳" in f7["text"]
    assert "Azure 2026-09 매출 +48%" in f7["text"]
    assert "nan" not in f7["text"]
    assert cloud.flags(h[h.ticker != "AMZN"], None)[0]["level"] == "보류"
    stale = h.copy()
    stale.loc[stale.ticker == "AMZN", "quarter_end"] = pd.Timestamp("2026-06-30")
    assert cloud.flags(stale, None)[0]["level"] == "보류"
