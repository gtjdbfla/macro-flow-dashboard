import pandas as pd
import pytest

from src.ai import end_demand as ed


def _ramp(values, end="2026-10-01"):
    idx = pd.date_range(end=end, periods=len(values), freq="MS")
    return pd.DataFrame({"전체": values}, index=idx)


def test_ramp_uses_latest_vintage_only(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "PARQUET", tmp_path)
    dates = pd.to_datetime(["2026-08-01", "2026-08-01", "2026-09-01"])
    pd.DataFrame({"date": dates, "available_at": pd.to_datetime(["2026-09-09", "2026-10-14", "2026-10-14"]),
                  "value": [62.5, 49.5, 50.0]}).to_parquet(tmp_path / "fredv_RAMPAISMB.parquet")
    r = ed.ramp("2026-11-01")
    assert r.loc["2026-08-01", "소기업"] == pytest.approx(0.495)
    assert ed.ramp("2026-10-01").loc["2026-08-01", "소기업"] == pytest.approx(0.625)


def test_adoption_levels_and_staleness():
    t = pd.DataFrame(columns=["recent", "accel", "gm_chg", "yoy", "yoy_prev", "name"])
    assert ed.flags(_ramp([0.50] * 6 + [0.54]), t, "2026-11-18")[0]["level"] == "긍정"
    assert ed.flags(_ramp([0.50] * 6 + [0.51]), t, "2026-11-18")[0]["level"] == "중립"
    assert ed.flags(_ramp([0.50] * 6 + [0.49]), t, "2026-11-18")[0]["level"] == "주의"
    assert ed.flags(_ramp([0.50] * 6 + [0.54]), t, "2027-01-20")[0]["level"] == "보류"


def _sw(accels, gms):
    return pd.DataFrame({"name": [f"c{i}" for i in range(len(accels))], "recent": True, "accel": accels, "gm_chg": gms,
                         "yoy": 0.2, "yoy_prev": 0.2})


def test_software_medians_use_each_available_value():
    ramp = _ramp([0.5] * 7)
    accels = [-0.07, -0.04, -0.01, 0.0, 0.01, 0.02, 0.03, 0.07, 0.07, 0.33]
    gms = [None, None, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.03, 0.01]
    assert ed.flags(ramp, _sw(accels, gms), "2026-11-18")[1]["level"] == "중립"
    assert ed.flags(ramp, _sw([0.03] * 10, [0.0] * 10), "2026-11-18")[1]["level"] == "긍정"
    assert ed.flags(ramp, _sw([0.03] * 10, [-0.03] * 10), "2026-11-18")[1]["level"] == "주의"
    assert ed.flags(ramp, _sw([-0.03] * 10, [0.0] * 10), "2026-11-18")[1]["level"] == "둔화"
    assert ed.flags(ramp, _sw([0.03] * 6, [0.0] * 6), "2026-11-18")[1]["level"] == "보류"


def test_software_yoy_compares_same_company_year_ago():
    ends = pd.to_datetime(["2024-07-31", "2025-07-31", "2026-07-31"])
    rows = [("CRM", e, "revenue", v, e + pd.Timedelta(days=30)) for e, v in zip(ends, [100.0, 110.0, 132.0])]
    rows += [("CRM", e, "gross_profit", v, e + pd.Timedelta(days=30)) for e, v in zip(ends, [70.0, 77.0, 89.1])]
    sec = pd.DataFrame(rows, columns=["ticker", "end", "metric", "value", "filed"])
    t, _ = ed.software(sec, "2026-11-18")
    crm = t.set_index("ticker").loc["CRM"]
    assert crm.yoy == pytest.approx(0.2)
    assert crm.yoy_prev == pytest.approx(0.1)
    assert crm.accel == pytest.approx(0.1)
    assert crm.gm_chg == pytest.approx(0.675 - 0.7)


def test_token_price_index_chains_only_same_models():
    day1, day2 = pd.Timestamp("2026-11-18"), pd.Timestamp("2026-11-19")
    df = pd.DataFrame([
        ("a/x", day1, 1.0, 10.0), ("a/y", day1, 1.0, 20.0), ("a/x:free", day1, 0.0, 0.0),
        ("a/x", day2, 1.0, 5.0), ("a/y", day2, 1.0, 20.0), ("a/new", day2, 1.0, 1.0),
    ], columns=["id", "fetched", "prompt", "completion"])
    df["created"] = pd.Timestamp("2026-10-01")
    idx = ed.token_price_index(df).set_index("fetched")["index"]
    assert idx[day1] == 100
    assert idx[day2] == pytest.approx(100 * (0.5 * 1.0) ** 0.5)


def test_revenue_coverage_uses_same_quarter_capex():
    hist = pd.DataFrame({"ticker": ["MSFT", "MSFT", "AMZN"],
                         "period_end": pd.to_datetime(["2024-12-31", "2026-03-31", "2026-06-30"]),
                         "capex_ttm": [520e8, 740e8, 1000e8]})
    cov = ed.ai_revenue_coverage(hist).set_index(["ticker", "period_end"])
    assert cov.loc[("MSFT", pd.Timestamp("2026-03-31")), "coverage"] == pytest.approx(370 / 740)


def test_anthropic_quarters_ratios_and_totals():
    q = ed.anthropic_quarters().set_index("quarter_end")
    assert q.loc["2026-09-30", "compute_ratio"] == pytest.approx(9863 / 16956)
    assert q.loc["2026-06-30", "op_income"] > 0
    nine_months = q.loc["2026-01-01":"2026-09-30"]
    assert nine_months.revenue.sum() == 33242          # S-1 요약표의 2026년 1~9월 매출
    assert nine_months.op_income.sum() == -1260        # S-1 요약표의 2026년 1~9월 영업손실


def test_holdings_rpo_yoy_and_basket_names():
    from src.ai import holdings as hd
    ends = pd.to_datetime(["2025-09-30", "2026-09-30"])
    rows = [("VRT", e, "rpo", v, e + pd.Timedelta(days=30)) for e, v in zip(ends, [100.0, 150.0])]
    sec = pd.DataFrame(rows, columns=["ticker", "end", "metric", "value", "filed"])
    assert hd._rpo_yoy(sec, "VRT", pd.Timestamp("2026-11-19")) == pytest.approx(0.5)
    assert hd._rpo_yoy(sec, "ETN", pd.Timestamp("2026-11-19")) is None
    assert set(hd.NAMES) >= {t for _, g in hd.GROUPS for t in g}
