import pandas as pd

from src.ai import early_warning as EW
from src.ai.guide import early_changes


def _series(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)), dtype=float)


def test_run_length():
    assert EW.run_length(pd.Series([True, False, True, True])) == 2
    assert EW.run_length(pd.Series([True, False])) == 0


def test_trend_needs_three_days():
    base = [100.0] * 200
    below2 = _series(base + [90, 90])
    assert EW.trend_state(below2)["state"] == "보유"
    below3 = _series(base + [90, 90, 90])
    assert EW.trend_state(below3)["state"] == "이탈"
    back = _series(base + [90, 90, 90, 120, 120])
    assert EW.trend_state(back)["state"] == "이탈"
    back3 = _series(base + [90, 90, 90, 120, 120, 120])
    assert EW.trend_state(back3)["state"] == "보유"


def test_trend_short_history():
    assert EW.trend_state(_series([1.0] * 100))["state"] == "자료 부족"


def test_levels():
    assert EW.level_hy(0.5) == "red" and EW.level_hy(0.3) == "yellow" and EW.level_hy(0.2) == "green"
    assert EW.level_jpy(-5) == "red" and EW.level_jpy(-3) == "yellow" and EW.level_jpy(1) == "green"
    assert EW.level_cpi(0.3) == "red" and EW.level_cpi(0.25) == "yellow" and EW.level_cpi(0.1) == "green"
    assert EW.level_sahm(0.5) == "red" and EW.level_sahm(0.3) == "yellow" and EW.level_sahm(0.1) == "green"


def test_early_changes():
    cur = {"levels": {"EW1": "red"}, "trend": {"NVDA": "이탈"}}
    prev = {"levels": {"EW1": "green"}, "trend": {"NVDA": "보유"}}
    assert len(early_changes(cur, prev)) == 2
    assert early_changes(cur, None) == []


def _tb(spx_state, since="2020-01-01", **others):
    rows = [{"ticker": "^GSPC", "state": spx_state, "since": since}]
    rows += [{"ticker": k, "state": v, "since": None} for k, v in others.items()]
    return rows


def test_stage_order(monkeypatch):
    monkeypatch.setattr(EW, "_px", lambda t: _series([100.0] * 400))
    assert EW.stage(_tb("이탈"), 1, {})["code"] == 3
    assert EW.stage(_tb("이탈"), 5, {})["code"] == 4
    assert EW.stage(_tb("이탈"), 0, {"vix": {"run_ge30": 3}})["code"] == 4
    assert EW.stage(_tb("보유", QQQ="이탈"), 0, {})["code"] == 2
    assert EW.stage(_tb("보유"), 2, {})["code"] == 2
    assert EW.stage(_tb("보유", QQQ="보유"), 1, {})["code"] == 1
