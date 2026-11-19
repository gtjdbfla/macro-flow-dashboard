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
