import json

import pandas as pd

from src.ai import guide as G


def _big4(rows):
    return pd.DataFrame(rows, columns=["period_end", "capex_yoy", "capex_ocf", "fcf_ttm", "opinc_per_capex"]).assign(
        period_end=lambda d: pd.to_datetime(d.period_end))


def _snap(date, outlook="mixed", level="주의", rows=None, flags=None):
    rows = rows or [("2026-06-30", 0.75, 0.77, 150.0, 0.27), ("2026-09-30", 0.87, 0.91, 61.0, 0.26)]
    risk = {"as_of": date, "score": 0.6, "level": "보통", "spx": 7000.0, "vix": 18.0, "shock": False,
            "macro": {"outlook": outlook, "as_of": date,
                      "metrics": {"rates": {"state": "neutral", "value": 2.8, "change": 1.0}}}}
    flags = flags or [{"id": "F7", "level": "긍정"}, {"id": "F11", "level": level}, {"id": "F16", "level": "주의"}]
    return G.snapshot(date, risk, flags, _big4(rows))


def test_checklist_compares_last_two_quarters():
    items = {i["id"]: i for i in _snap("2026-11-19")["checklist"]}
    assert items["capex_decel"]["ok"] is False        # 0.75 -> 0.87 증가율이 더 올랐다
    assert items["capex_ocf_down"]["ok"] is False
    assert items["fcf_up"]["ok"] is False
    assert items["payback_up"]["ok"] is False
    assert items["cloud_ok"]["ok"] is True
    assert items["nvda_cash_ok"]["ok"] is False        # F16 주의
    assert items["borrowers_ok"]["ok"] is True         # F11 주의는 경고가 아니다


def test_checklist_marks_missing_data_not_false():
    items = {i["id"]: i for i in G.checklist(_big4([("2026-09-30", 0.87, 0.91, 61.0, 0.26)]), {})}
    assert items["capex_decel"]["ok"] is None
    assert items["cloud_ok"]["ok"] is None


def test_compare_reports_changes_against_a_week_ago(tmp_path):
    path = tmp_path / "guide.jsonl"
    G.append(path, _snap("2026-11-10", "pending", "주의"))
    G.append(path, _snap("2026-11-14", "mixed", "주의"))
    rows = G.append(path, _snap("2026-11-18", "caution", "경고"))
    changes = G.compare(rows)
    assert changes["base"] == "2026-11-10"
    assert {"id": "F11", "from": "주의", "to": "경고", "direction": "악화"} in changes["flag_changes"]
    assert changes["macro_changes"][0] == {"name": "종합 판정", "from": "pending", "to": "caution"}


def test_append_replaces_same_date(tmp_path):
    path = tmp_path / "guide.jsonl"
    G.append(path, _snap("2026-11-18", "mixed"))
    rows = G.append(path, _snap("2026-11-18", "caution"))
    assert len(rows) == 1 and rows[0]["macro"]["outlook"] == "caution"
    assert len(path.read_text().splitlines()) == 1
    assert json.loads(path.read_text().splitlines()[0])["date"] == "2026-11-18"


def test_compare_without_history_is_empty():
    assert G.compare([_snap("2026-11-18")])["base"] is None


def _ctx(**kw):
    base = {"real_rate_20d": 0.13, "hy_20d": 0.54, "hy_vs_low": 1.03, "support_ratio": [0.66, 0.55],
            "support_detail": "약정 1,985억 → 1,980억 달러, 12개월 매출 +19% (비교 2점)", "coverage": [("2024-12-31", 0.23), ("2026-03-31", 0.38)],
            "anthropic_ratio": [0.716, 0.481, 0.582], "crwv_cds": [("2026-11-05", 1289), ("2026-11-06", 1155)], "soxx_1m": -0.19, "vix": 28.4}
    base.update(kw)
    return base


def _bubble(ctx, flags=None, rows=None, today="2026-11-20"):
    rows = rows or [("2026-06-30", 0.75, 0.77, 150e8, 0.27), ("2026-09-30", 0.87, 0.91, 61e8, 0.26)]
    big4 = _big4(rows).assign(ocf_yoy=[0.12, 0.10][: len(rows)] if len(rows) == 2 else 0.1)
    flags = flags or {"F7": "긍정", "F11": "경고", "F15": "보류", "F4": "긍정", "F8": "강함", "F16": "주의", "F17": "긍정", "F18": "중립"}
    return G.bubble_checks(big4, flags, ctx, today)


def _by_id(b):
    return {i["id"]: i for i in b["resolution"] + b["warning"]}


def test_bubble_resolution_items_follow_direction_of_each_condition():
    b = _by_id(_bubble(_ctx()))
    assert "A3" not in b                                                               # A1·A3 통합
    assert b["A1"]["ok"] is False and b["A2"]["ok"] is False                           # 77% -> 91%
    assert b["B1"]["ok"] is False                                                      # F18 중립
    assert b["B2"]["ok"] is None                                                       # 최신 공개(2026-03)가 200일보다 오래됨
    assert b["B3"]["ok"] is False                                                      # 71.6% -> 48.1% -> 58.2%: 2분기 연속 하락 아님
    assert b["C1"]["ok"] is True and "약정 1,985억" in b["C1"]["detail"]               # 66% -> 55%
    assert b["C4"]["ok"] is None                                                       # 최신 기록이 14일 전
    assert b["C5"]["ok"] is False and b["C6"]["ok"] is False
    assert b["D1"]["ok"] is False and b["D2"]["ok"] is False


def test_a1_accepts_level_or_five_point_drop():
    ok_level = _by_id(_bubble(_ctx(), rows=[("2026-06-30", 0.5, 0.80, 100e8, 0.2), ("2026-09-30", 0.5, 0.74, 100e8, 0.2)]))
    ok_drop = _by_id(_bubble(_ctx(), rows=[("2026-06-30", 0.5, 0.95, 100e8, 0.2), ("2026-09-30", 0.5, 0.89, 100e8, 0.2)]))
    bad = _by_id(_bubble(_ctx(), rows=[("2026-06-30", 0.5, 0.90, 100e8, 0.2), ("2026-09-30", 0.5, 0.87, 100e8, 0.2)]))
    assert ok_level["A1"]["ok"] is True and ok_drop["A1"]["ok"] is True and bad["A1"]["ok"] is False


def test_bubble_staleness_and_improving_values():
    ctx = _ctx(coverage=[("2026-03-31", 0.38), ("2026-09-30", 0.5)], crwv_cds=[("2026-11-15", 1200), ("2026-11-18", 900)],
               real_rate_20d=-0.3, hy_20d=-0.25, soxx_1m=0.03, vix=20.0, anthropic_ratio=[0.7, 0.55, 0.5])
    b = _by_id(_bubble(ctx))
    assert b["B2"]["ok"] is True and b["C4"]["ok"] is True and b["B3"]["ok"] is True
    assert b["C5"]["ok"] is True and b["C6"]["ok"] is True
    assert b["D1"]["ok"] is True and b["D2"]["ok"] is True
    small = _by_id(_bubble(_ctx(real_rate_20d=-0.1, hy_20d=-0.1)))
    assert small["C5"]["ok"] is False and small["C6"]["ok"] is False                   # 임계값(25bp, 20bp)에 못 미치면 충족 아님


def test_bubble_warning_signals_and_bad_slowdown():
    flags = {"F7": "중립", "F4": "중립", "F8": "보통", "F15": "경고"}
    rows = [("2026-06-30", 0.90, 0.77, 150e8, 0.27), ("2026-09-30", 0.60, 0.91, 61e8, 0.26)]   # CAPEX 증가율 -30%p
    b = _by_id(_bubble(_ctx(), flags, rows))
    assert all(b[k]["ok"] is True for k in ("W1", "W2", "W3", "W4", "W5"))
    healthy = _by_id(_bubble(_ctx(), {"F7": "긍정", "F4": "긍정", "F8": "강함", "F15": "정상"}, rows))
    assert healthy["W5"]["ok"] is False and healthy["W1"]["ok"] is False


def test_new_financing_warnings():
    rows5 = [("2025-09-30", 0.5, 0.60, 2000e8, 0.2), ("2025-12-31", 0.5, 0.65, 2000e8, 0.2), ("2026-03-31", 0.5, 0.70, 1800e8, 0.2),
             ("2026-06-30", 0.5, 0.77, 1500e8, 0.2), ("2026-09-30", 0.5, 1.02, 600e8, 0.2)]
    b = _by_id(_bubble(_ctx(anthropic_ratio=[0.40, 0.48, 0.58], crwv_cds=[("2026-11-05", 1100), ("2026-11-18", 1100)]), rows=rows5))
    assert b["W6"]["ok"] is True and b["W7"]["ok"] is True and b["W8"]["ok"] is True and b["W9"]["ok"] is True and b["W10"]["ok"] is True
    calm = _by_id(_bubble(_ctx(hy_vs_low=0.4, anthropic_ratio=[0.6, 0.5, 0.45], crwv_cds=[("2026-11-05", 600), ("2026-11-18", 600)]),
                          rows=[("2025-09-30", 0.5, 0.40, 1500e8, 0.2), ("2025-12-31", 0.5, 0.40, 1600e8, 0.2), ("2026-03-31", 0.5, 0.40, 1700e8, 0.2),
                                ("2026-06-30", 0.5, 0.40, 1800e8, 0.2), ("2026-09-30", 0.5, 0.40, 1900e8, 0.2)]))
    assert not any(calm[k]["ok"] for k in ("W6", "W7", "W8", "W9", "W10"))
    sparse = _by_id(_bubble({}))
    assert sparse["W8"]["ok"] is None and sparse["W9"]["ok"] is None and sparse["W10"]["ok"] is None and sparse["W6"]["ok"] is None


def test_bubble_missing_inputs_are_not_false():
    b = _by_id(_bubble({}))
    assert b["C1"]["ok"] is None and b["C5"]["ok"] is None and b["D2"]["ok"] is None and b["B3"]["ok"] is None


def test_bubble_changes_between_snapshots():
    before = _bubble(_ctx(real_rate_20d=0.13))
    after = _bubble(_ctx(real_rate_20d=-0.30))
    changes = G.bubble_changes(after, before)
    assert [c["id"] for c in changes] == ["C5"] and changes[0]["from"] is False and changes[0]["to"] is True
    assert G.bubble_changes(after, None) == []


def test_annotate_since_marks_exact_record_and_new():
    from src.ai.guide import annotate_since
    def snap(date, a, b=None):
        res = [{"id": "A1", "ok": a}] + ([{"id": "B5", "ok": b}] if b is not None else [])
        return {"date": date, "bubble": {"resolution": res, "warning": []}}
    rows = [snap("2026-11-20", False), snap("2026-11-21", True), snap("2026-11-22", True, True)]
    cur = {"resolution": [{"id": "A1", "ok": True}, {"id": "B5", "ok": True}], "warning": []}
    annotate_since(cur, rows)
    assert (cur["resolution"][0]["since"], cur["resolution"][0]["since_kind"]) == ("2026-11-21", "exact")
    assert (cur["resolution"][1]["since"], cur["resolution"][1]["since_kind"]) == ("2026-11-22", "new")
    rows2 = [snap("2026-11-20", True), snap("2026-11-21", True)]
    cur2 = {"resolution": [{"id": "A1", "ok": True}], "warning": []}
    annotate_since(cur2, rows2)
    assert (cur2["resolution"][0]["since"], cur2["resolution"][0]["since_kind"]) == ("2026-11-20", "record")


def test_retro_since_ramp_and_nvidia_support():
    from src.ai.guide import retro_since, streak_start
    idx = pd.date_range("2025-01-01", periods=12, freq="MS")
    ramp = pd.Series([.30, .31, .32, .33, .34, .35, .36, .38, .40, .42, .44, .46], index=idx)   # 6개월 변화: 7월부터 +6%p 이상
    sup = pd.Series([float("nan"), 0.655, 0.547], index=pd.to_datetime(["2026-04-26", "2026-07-26", "2026-10-25"]))
    b = {"resolution": [{"id": "B4", "ok": True}, {"id": "C1", "ok": True}], "warning": []}
    retro_since(b, ramp_all=ramp, support=sup, released={"2026-10-25": "2026-11-17"})
    assert b["resolution"][0]["since_kind"] == "obs" and b["resolution"][0]["since"] == "2025-07-01"
    assert (b["resolution"][1]["since"], b["resolution"][1]["since_kind"]) == ("2026-11-17", "release")
    assert streak_start(pd.Series([3, 2, 1]), lambda c, p: c < p) == 1 or streak_start(pd.Series([3, 2, 1]), lambda c, p: c < p) == 2
