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
    base = {"real_rate_3m": 0.47, "hy_4w": 0.17, "support_ratio": [0.66, 0.55], "coverage": [("2024-12-31", 0.23), ("2026-03-31", 0.38)],
            "anthropic_ratio": [0.48, 0.58], "crwv_cds": [("2026-11-05", 1289), ("2026-11-06", 1155)], "soxx_1m": -0.19, "vix": 28.4}
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
    assert b["A1"]["ok"] is False and b["A2"]["ok"] is False and b["A3"]["ok"] is False   # CAPEX 87% > 영업현금흐름 10%
    assert b["B1"]["ok"] is False                                                      # F18 중립
    assert b["B2"]["ok"] is None                                                       # 최신 공개(2026-03)가 200일보다 오래됨
    assert b["B3"]["ok"] is False                                                      # 48% -> 58% 상승
    assert b["C1"]["ok"] is True                                                       # 66% -> 55% 하락
    assert b["C4"]["ok"] is None                                                       # 최신 기록이 14일 전
    assert b["C5"]["ok"] is False and b["C6"]["ok"] is False
    assert b["D1"]["ok"] is False and b["D2"]["ok"] is False


def test_bubble_staleness_and_improving_values():
    ctx = _ctx(coverage=[("2026-03-31", 0.38), ("2026-09-30", 0.5)], crwv_cds=[("2026-11-15", 1200), ("2026-11-18", 900)],
               real_rate_3m=-0.1, hy_4w=-0.2, soxx_1m=0.03, vix=20.0)
    b = _by_id(_bubble(ctx))
    assert b["B2"]["ok"] is True and b["C4"]["ok"] is True
    assert b["C5"]["ok"] is True and b["C6"]["ok"] is True
    assert b["D1"]["ok"] is True and b["D2"]["ok"] is True


def test_bubble_warning_signals_and_bad_slowdown():
    flags = {"F7": "중립", "F4": "중립", "F8": "보통", "F15": "경고"}
    rows = [("2026-06-30", 0.90, 0.77, 150e8, 0.27), ("2026-09-30", 0.60, 0.91, 61e8, 0.26)]   # CAPEX 증가율 -30%p
    b = _by_id(_bubble(_ctx(), flags, rows))
    assert all(b[k]["ok"] is True for k in ("W1", "W2", "W3", "W4", "W5"))
    healthy = _by_id(_bubble(_ctx(), {"F7": "긍정", "F4": "긍정", "F8": "강함", "F15": "정상"}, rows))
    assert healthy["W5"]["ok"] is False and healthy["W1"]["ok"] is False


def test_bubble_missing_inputs_are_not_false():
    b = _by_id(_bubble({}))
    assert b["C1"]["ok"] is None and b["C5"]["ok"] is None and b["D2"]["ok"] is None


def test_bubble_changes_between_snapshots():
    before = _bubble(_ctx(real_rate_3m=0.47))
    after = _bubble(_ctx(real_rate_3m=-0.05))
    changes = G.bubble_changes(after, before)
    assert [c["id"] for c in changes] == ["C5"] and changes[0]["from"] is False and changes[0]["to"] is True
    assert G.bubble_changes(after, None) == []
