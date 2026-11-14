"""운영 판정의 공개 시점과 데이터 누락을 검증한다."""
import pandas as pd

from scoring import macro_outlook as mo


def test_outlook_requires_complete_inputs_and_market_confirmation():
    metrics = {key: {"state": "neutral"} for key in mo.METRICS}
    metrics["credit"]["state"] = "adverse"
    metrics["liquidity"]["state"] = "adverse"
    assert mo.outlook(metrics, "보통", False, True) == "mixed"
    assert mo.outlook(metrics, "보통", True, True) == "caution"
    assert mo.outlook(metrics, "높음", False, True) == "caution"
    assert mo.outlook(metrics, "높음", False, False) == "pending"
    metrics["labor"]["state"] = "stale"
    assert mo.outlook(metrics, "높음", True, True) == "pending"


def test_favorable_outlook_requires_no_market_stress():
    metrics = {key: {"state": "supportive"} for key in mo.METRICS}
    assert mo.outlook(metrics, "낮음", False, True) == "supportive"
    assert mo.outlook(metrics, "높음", False, True) == "mixed"
    metrics["credit"]["state"] = "adverse"
    assert mo.outlook(metrics, "낮음", False, True) == "mixed"


def test_snapshot_uses_only_available_observations_and_keeps_missing_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "PARQUET", tmp_path)
    dates = pd.bdate_range(end="2026-11-05", periods=22)
    pd.DataFrame({"value": [3.0] * 20 + [2.75, 7.0]}, index=dates).to_parquet(tmp_path / "fred_DFII10.parquet")
    dates = pd.date_range(end="2026-10-31", periods=10, freq="W-SAT")
    claims = pd.DataFrame({"date": dates, "available_at": dates + pd.Timedelta(days=5),
                           "value": [200000.0] * 9 + [800000.0]})
    claims.to_parquet(tmp_path / "fredv_ICSA.parquet")

    snapshot = mo.snapshot("2026-11-04", "보통", False, True)
    assert snapshot["outlook"] == "pending"
    assert snapshot["metrics"]["rates"]["as_of"] == "2026-11-03"
    assert snapshot["metrics"]["rates"]["state"] == "missing"
    assert snapshot["metrics"]["labor"]["as_of"] == "2026-10-24"
    assert snapshot["metrics"]["labor"]["value"] == 200000
    assert snapshot["metrics"]["credit"]["state"] == "missing"

    snapshot = mo.snapshot("2026-11-05", "보통", False, True)
    assert snapshot["metrics"]["rates"]["change"] == -25
    assert snapshot["metrics"]["rates"]["state"] == "supportive"
    assert snapshot["metrics"]["labor"]["as_of"] == "2026-10-31"

    snapshot = mo.snapshot("2026-11-23", "보통", False, True)
    assert snapshot["metrics"]["rates"]["state"] == "stale"
    assert snapshot["metrics"]["labor"]["state"] == "stale"


def test_inflation_falls_back_to_last_month_with_a_year_over_year_value(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "PARQUET", tmp_path)
    months = pd.date_range("2024-01-01", "2026-10-01", freq="MS")
    values = pd.Series(range(len(months)), index=months, dtype=float) + 300
    values = values.drop(pd.Timestamp("2025-10-01"))   # 전년 같은 달 값이 없으면 2026-10 전년비를 못 구한다
    cpi = pd.DataFrame({"date": values.index, "available_at": values.index + pd.Timedelta(days=45),
                        "value": values.to_numpy()})
    cpi.to_parquet(tmp_path / "fredv_CPILFESL.parquet")

    inflation = mo.snapshot("2026-11-12", "보통", False, True)["metrics"]["inflation"]
    assert inflation["as_of"] == "2026-09-01"
    assert inflation["state"] in {"supportive", "neutral", "adverse"}


def test_liquidity_uses_previous_business_day_repo_when_wednesday_is_a_holiday(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "PARQUET", tmp_path)
    weeks = pd.date_range("2026-10-07", "2026-11-11", freq="W-WED")
    pd.DataFrame({"value": [7_000_000.0] * len(weeks)}, index=weeks).to_parquet(tmp_path / "fred_WALCL.parquet")
    pd.DataFrame({"value": [900_000.0] * len(weeks)}, index=weeks).to_parquet(tmp_path / "fred_WTREGEN.parquet")
    days = pd.bdate_range("2026-10-07", "2026-11-13").drop(pd.Timestamp("2026-11-11"))
    pd.DataFrame({"value": [10.0] * len(days)}, index=days).to_parquet(tmp_path / "fred_RRPONTSYD.parquet")

    liquidity = mo.snapshot("2026-11-13", "보통", False, True)["metrics"]["liquidity"]
    assert liquidity["as_of"] == "2026-11-11"
    assert liquidity["value"] == 6090.0
