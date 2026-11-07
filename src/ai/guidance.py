"""공식 실적 발표에서 확인한 가이던스와 계약잔고. 금액 입력 단위는 억 달러."""
from dataclasses import dataclass

import pandas as pd

MSFT_JUL = "https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q4"
MSFT_OCT = "https://www.microsoft.com/en-us/investor/events/fy-2027/earnings-fy-2027-q1"
MSFT_ACTUAL = "https://www.microsoft.com/en-us/Investor/earnings/FY-2027-Q1/press-release-webcast"
MSFT_10Q = "https://www.sec.gov/Archives/edgar/data/789019/000119312526436133/msft-20260930.htm"
AMZN_JUL = "https://ir.aboutamazon.com/news-release/news-release-details/2026/Amazon-com-Announces-Second-Quarter-Results/default.aspx"
AMZN_OCT = "https://ir.aboutamazon.com/news-release/news-release-details/2026/Amazon-com-Announces-Third-Quarter-Results/default.aspx"
META_JUL = "https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Second-Quarter-2026-Results/default.aspx"
META_OCT = "https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Third-Quarter-2026-Results/default.aspx"
ORCL_JUN = "https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Record-Q4-and-FY-2026-Results-Driven-by-Cloud-Infrastructure--Cloud-Applications/default.aspx"
ORCL_SEP = "https://investor.oracle.com/investor-news/news-details/2026/Oracle-Announces-Q1-Results-Driven-by-Triple-Digit-Growth-in-Cloud-Infrastructure-Revenues/default.aspx"
ORCL_10Q = "https://www.sec.gov/Archives/edgar/data/1341439/000119312526389274/orcl-20260831.htm"
GOOGL_10Q = "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000084/goog-20260930.htm"
CRWV_10Q = "https://www.sec.gov/Archives/edgar/data/1769628/000176962826000366/crwv-20260630.htm"

NAMES = {"MSFT": "Microsoft", "ORCL": "Oracle", "AMZN": "Amazon", "META": "Meta"}
CIK = {"MSFT": 789019, "ORCL": 1341439, "AMZN": 1018724, "META": 1326801}
AZURE = "Azure 매출 전년비 · 환율 고정"
ORACLE_CLOUD = "전체 클라우드 매출 전년비 · USD"


@dataclass(frozen=True)
class Guide:
    ticker: str
    period: str
    issued: str
    metric: str
    unit: str
    low: float
    high: float
    source: str
    sec_metric: str | None = None
    actual: float | None = None
    actual_source: str | None = None


GUIDES = (
    Guide("MSFT", "2026-09-30", "2026-07-29", AZURE, "%", 45, 45, MSFT_JUL,
          actual=48, actual_source=MSFT_ACTUAL),
    Guide("ORCL", "2026-08-31", "2026-06-10", ORACLE_CLOUD, "%", 58, 64, ORCL_JUN,
          actual=62, actual_source=ORCL_SEP),
    Guide("AMZN", "2026-09-30", "2026-07-30", "회사 전체 매출", "억 달러", 1970, 2020, AMZN_JUL,
          sec_metric="revenue", actual=2019.44, actual_source=AMZN_OCT),
    Guide("AMZN", "2026-09-30", "2026-07-30", "회사 전체 영업이익", "억 달러", 225, 265, AMZN_JUL,
          sec_metric="op_income", actual=280.76, actual_source=AMZN_OCT),
    Guide("META", "2026-09-30", "2026-07-29", "회사 전체 매출", "억 달러", 610, 640, META_JUL,
          sec_metric="revenue", actual=646.01, actual_source=META_OCT),
    Guide("MSFT", "2026-12-31", "2026-10-28", AZURE, "%", 49, 50, MSFT_OCT),
    Guide("MSFT", "2026-12-31", "2026-10-28", "Microsoft Cloud 매출총이익률", "%", 65, 65, MSFT_OCT),
    Guide("ORCL", "2026-11-30", "2026-09-10", ORACLE_CLOUD, "%", 65, 71, ORCL_SEP),
    Guide("AMZN", "2026-12-31", "2026-10-29", "회사 전체 매출", "억 달러", 2400, 2470, AMZN_OCT,
          sec_metric="revenue"),
    Guide("AMZN", "2026-12-31", "2026-10-29", "회사 전체 영업이익", "억 달러", 285, 335, AMZN_OCT,
          sec_metric="op_income"),
    Guide("META", "2026-12-31", "2026-10-21", "회사 전체 매출", "억 달러", 730, 760, META_OCT,
          sec_metric="revenue"),
)

# 서로 다른 회사의 인식 기간이나 계약 범위는 합산하지 않는다.
BACKLOG = (
    ("Microsoft", "2026-09-30", "상업 부문 RPO", 6770, 12, "약 30%", .30, MSFT_10Q),
    ("Oracle", "2026-08-31", "회사 전체 RPO", 6640, 12, "약 13%", .13, ORCL_10Q),
    ("Alphabet", "2026-09-30", "Google Cloud 계약잔고", 5608, 24, "55% 조금 넘음", .55, GOOGL_10Q),
    ("CoreWeave", "2026-06-30", "회사 전체 RPO", 1037, 24, "41%", .41, CRWV_10Q),
)


def _quarterly_values(sec):
    if sec is None or sec.empty:
        return {}
    dates = {pd.Timestamp(row.period) for row in GUIDES}
    d = sec.loc[sec.ticker.isin(NAMES) & sec.end.isin(dates)]
    d = d.sort_values("filed").drop_duplicates(["ticker", "end", "metric"], keep="first")
    return d.set_index(["ticker", "end", "metric"])["value"].dropna().to_dict()


def _judgment(guide, actual):
    if guide.low == guide.high:
        unit = "%p" if guide.unit == "%" else guide.unit
        return f"제시치 대비 {actual - guide.low:+.1f}{unit}"
    if actual > guide.high:
        return "범위 상회"
    if actual < guide.low:
        return "범위 하회"
    return "범위 안"


def _number(value):
    return f"{value:,.1f}".removesuffix(".0")


def comparisons(sec):
    values = _quarterly_values(sec)
    rows = []
    for guide in GUIDES:
        key = (guide.ticker, pd.Timestamp(guide.period))
        actual = guide.actual
        if guide.sec_metric:
            observed = values.get((*key, guide.sec_metric))
            if observed is not None:
                actual = observed / 1e8
        ocf, capex = values.get((*key, "ocf")), values.get((*key, "capex"))
        fcf = (ocf - capex) / 1e8 if ocf is not None and capex is not None else None
        expected = f"약 {_number(guide.low)}" if guide.low == guide.high else f"{_number(guide.low)}~{_number(guide.high)}"
        sep = "" if guide.unit == "%" else " "
        rows.append({
            "회사": NAMES[guide.ticker], "대상 분기말": guide.period, "지표": guide.metric,
            "회사 제시": f"{expected}{sep}{guide.unit}",
            "실제": f"{_number(actual)}{sep}{guide.unit}" if actual is not None else None,
            "결과": _judgment(guide, actual) if actual is not None else (
                "같은 지표 확인 대기" if values.get((*key, "revenue")) is not None else "실적 확인 대기"
            ),
            "같은 분기 회사 전체 FCF (억 달러)": fcf, "제시일": guide.issued, "제시 원문": guide.source,
            "실적 원문": (guide.actual_source or f"https://www.sec.gov/edgar/browse/?CIK={CIK[guide.ticker]}")
            if actual is not None else None,
        })
    return pd.DataFrame(rows)
