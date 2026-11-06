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
