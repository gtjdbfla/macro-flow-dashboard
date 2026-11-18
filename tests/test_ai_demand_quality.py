import pandas as pd
import pytest

from src.ai import demand_quality as dq


def _sec(rows):
    """rows: (분기말, 매출, 매출채권, 재고, 영업현금흐름, 순이익), 금액 단위 억 달러."""
    out = []
    for end, *values in rows:
        for metric, value in zip(["revenue", "ar", "inventory", "ocf", "net_income"], values):
            if value is not None:
                out.append(("NVDA", pd.Timestamp(end), metric, value * 1e8, pd.Timestamp(end) + pd.Timedelta(days=30)))
    return pd.DataFrame(out, columns=["ticker", "end", "metric", "value", "filed"])


QUARTERS = ["2025-04-27", "2025-07-27", "2025-10-26", "2026-01-25", "2026-04-26", "2026-07-26"]


def test_dso_uses_quarter_days_and_compares_with_year_ago():
    sec = _sec([(q, 100, 50, 100, 90, 100) for q in QUARTERS[:5]] + [(QUARTERS[5], 100, 70, 100, 90, 100)])
    t = dq.table(sec)
    assert t.at[pd.Timestamp("2026-07-26"), "dso"] == pytest.approx(70 / 100 * 91)
    assert t.at[pd.Timestamp("2026-07-26"), "dso_chg"] == pytest.approx(20 / 100 * 91)
    assert t.at[pd.Timestamp("2026-07-26"), "inv_gap"] == pytest.approx(0)


def test_levels_count_deteriorating_checks(monkeypatch):
    monkeypatch.setattr(dq, "COMMITMENTS", ())
    healthy = _sec([(q, 100, 50, 100, 90, 100) for q in QUARTERS])
    assert dq.flags(dq.table(healthy))[0]["level"] == "정상"
    one = _sec([(q, 100, 50, 100, 70, 100) for q in QUARTERS])        # 현금 전환 0.7
    assert dq.flags(dq.table(one))[0]["level"] == "주의"
    two = _sec([(q, 100, 50, 100, 70, 100) for q in QUARTERS[:5]] + [(QUARTERS[5], 100, 70, 150, 70, 100)])
    assert dq.flags(dq.table(two))[0]["level"] == "경고"


def test_support_ratio_uses_release_revenue_before_10q(monkeypatch):
    monkeypatch.setattr(dq, "COMMITMENTS", (
        dq.Commitments("2026-07-26", 100, 0, 0, 0, 0, 0, 100, "a"),
        dq.Commitments("2026-10-25", 200, 0, 0, 0, 0, 0, 100, "b"),
    ))
    t = dq.table(_sec([(q, 100, 50, 100, 90, 100) for q in QUARTERS]))
    assert t.at[pd.Timestamp("2026-10-25"), "revenue_ttm"] == pytest.approx(400e8)
    assert t.at[pd.Timestamp("2026-10-25"), "support_ratio"] == pytest.approx(0.5)
    flag = dq.flags(t)[0]
    assert "D 고객 지원성 약정 ÷ 매출 50%(직전 분기 대비 +25%p" in flag["text"]
    assert flag["level"] == "주의"


def test_pending_when_fewer_than_three_checks(monkeypatch):
    monkeypatch.setattr(dq, "COMMITMENTS", ())
    t = dq.table(_sec([(q, 100, 50, 100, 90, 100) for q in QUARTERS[:4]]))
    assert dq.flags(t)[0]["level"] == "보류"


def test_platform_share_is_within_data_center_customers():
    mix = dq.platform()
    assert mix.period_end.is_monotonic_increasing
    last = mix.iloc[-1]
    assert last.ai_clouds_share == pytest.approx(475 / (601 + 475))
