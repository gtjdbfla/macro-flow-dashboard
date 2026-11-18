import pandas as pd
import pytest

from src.ai.guidance import comparisons


def test_guidance_matches_period_metric_and_company():
    sec = pd.DataFrame([
        ("AMZN", "2026-12-31", "op_income", 32e9, "2027-01-29"),
        ("AMZN", "2026-12-31", "op_income", 40e9, "2027-02-01"),
        ("AMZN", "2026-12-31", "ocf", 60e9, "2027-01-29"),
        ("AMZN", "2026-09-30", "capex", 40e9, "2026-10-30"),
        ("MSFT", "2026-12-31", "revenue", 96e9, "2027-01-29"),
        ("META", "2026-09-30", "ocf", 32.785e9, "2026-10-22"),
        ("META", "2026-09-30", "capex", 33.942e9, "2026-10-22"),
    ], columns=["ticker", "end", "metric", "value", "filed"])
    sec["end"] = pd.to_datetime(sec.end)
    rows = comparisons(sec).set_index(["회사", "대상 분기말", "지표"])
    amazon = rows.loc[("Amazon", "2026-12-31", "회사 전체 영업이익")]
    assert amazon["실제"] == "320 억 달러"
    assert amazon["결과"] == "범위 안"
    assert pd.isna(amazon["같은 분기 회사 전체 FCF (억 달러)"])
    azure = rows.loc[("Microsoft", "2026-12-31", "Azure 매출 전년비 · 환율 고정")]
    assert pd.isna(azure["실제"])
    assert azure["결과"] == "같은 지표 확인 대기"
    meta = rows.loc[("Meta", "2026-09-30", "회사 전체 매출")]
    assert meta["결과"] == "범위 상회"
    assert meta["같은 분기 회사 전체 FCF (억 달러)"] == pytest.approx(-11.57)


def test_nvidia_guidance_uses_release_actuals_until_10q():
    rows = comparisons(pd.DataFrame(columns=["ticker", "end", "metric", "value", "filed"])).set_index(["회사", "대상 분기말", "지표"])
    revenue = rows.loc[("NVIDIA", "2026-10-25", "회사 전체 매출")]
    assert revenue["결과"] == "범위 상회"
    margin = rows.loc[("NVIDIA", "2026-10-25", "GAAP 매출총이익률")]
    assert margin["결과"] == "범위 상회"
    assert rows.loc[("NVIDIA", "2027-01-31", "회사 전체 매출")]["결과"] == "실적 확인 대기"
