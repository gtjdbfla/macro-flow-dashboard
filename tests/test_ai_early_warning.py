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


def test_scorecard_matures_and_counts():
    from src.ai import scorecard as SC
    idx = pd.bdate_range("2027-01-01", periods=100)
    price = pd.Series(100.0, index=idx)
    price.iloc[30:] = 85.0                       # 30일째부터 −15%
    rows = [{"date": str(idx[0].date()), "early": {"levels": {"EW3": "red"}, "trend": {"QQQ": "보유"}, "stage": "주의"}},
            {"date": str(idx[50].date()), "early": {"levels": {"EW3": "green"}, "trend": {"QQQ": "이탈"}, "stage": "경계"}}]
    out = SC.score(rows, price)
    assert out["matured"] == 1                   # 두 번째는 63거래일이 안 지남
    assert out["signals"]["EW3 빨강"] == {"on": 1, "hit": 1, "rate": 1.0}
    assert out["signals"]["단계 경계 이상"]["on"] == 0
    assert out["base_rate"] == 1.0 and not out["enough"]


def test_bubble_b5_cloud_gain():
    from src.ai.guide import bubble_checks
    big4 = pd.DataFrame({"period_end": pd.to_datetime(["2026-06-30", "2026-09-30"]), "capex_yoy": [0.5, 0.6], "capex_ocf": [0.8, 0.9],
                         "fcf_ttm": [2e11, 1e11], "ocf_yoy": [0.2, 0.2]})
    up = {"AMZN": [("2026-06-30", 0.3), ("2026-09-30", 0.4)], "GOOGL": [("2026-06-30", 0.5), ("2026-09-30", 0.8)]}
    b = lambda cg: {i["id"]: i for i in bubble_checks(big4, {}, {"cloud_gain": cg}, "2026-11-25")["resolution"]}["B5"]
    assert b(up)["ok"] is True
    assert b({**up, "AMZN": [("2026-06-30", 0.5), ("2026-09-30", 0.4)]})["ok"] is False
    assert b({"AMZN": up["AMZN"]})["ok"] is None


def test_circular_facts_and_amazon_share():
    from src.ai import circular as CR
    d = CR.facts()
    assert set(d.columns) == {"고리", "기준", "항목", "내용", "출처"} and len(d) >= 10
    assert d.출처.str.startswith("https://").all()
    assert abs(CR.amazon_markup_share() - 0.4256) < 0.001


def test_audit_freshness_cadence_rules():
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_freshness", "scripts/audit_freshness.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.max_age("fred_DGS10") == 6 and m.max_age("fred_WALCL") == 14 and m.max_age("fredv_CPILFESL") == 80
    assert m.max_age("fredv_CP") == 150 and m.max_age("french_ff") == 75 and m.max_age("kis_investor_kospi") == 6
    assert m.max_age("ai_history") is None


def test_ew10_capex_guide_needs_two_companies_and_flags_below_20():
    assert EW._capex_guide(())["level"] == "na"
    one = (("MSFT", "2027-01-28", "FY27", 1000, 30.0, "https://x", ""),)
    assert EW._capex_guide(one)["level"] == "na"
    two = one + (("GOOGL", "2027-02-04", "2027", 900, 10.0, "https://y", ""),)
    assert EW._capex_guide(two)["level"] == "green"         # 중앙값 (30+10)/2=20은 '20% 미만'이 아니다
    three = two + (("META", "2027-01-29", "2027", 800, 5.0, "https://z", ""),)
    assert EW._capex_guide(three)["level"] == "red"
    ok = (("MSFT", "2027-01-28", "FY27", 1000, 40.0, "https://x", ""), ("GOOGL", "2027-02-04", "2027", 900, 25.0, "https://y", ""))
    assert EW._capex_guide(ok)["level"] == "green"
