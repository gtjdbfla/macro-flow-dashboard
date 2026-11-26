"""F16 수요의 질(NVIDIA): 매출이 현금으로 들어오는지, 고객의 자금 조달을 얼마나 떠받치는지. 규칙은 docs/AI_PROFITABILITY.md에 사전 등록."""
from dataclasses import dataclass

import pandas as pd

DSO_UP, INV_GAP, CASH_CONV, SUPPORT_UP = 10, 0.20, 0.8, 0.10

Q2FY27 = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000073/q2fy27cfocommentary.htm"
Q3FY27 = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000088/q3fy27cfocommentary.htm"


@dataclass(frozen=True)
class Commitments:
    """CFO 코멘터리의 약정·보증. 금액 입력 단위는 억 달러. revenue_q는 10-Q 전에 판정하기 위한 실적 발표 분기 매출."""
    period: str
    guarantees: float     # 보증 최대 노출
    neocloud: float       # 네오클라우드(구 AI cloud) 약정
    equity: float         # 지분투자 약정
    cloud_service: float  # 클라우드 서비스 약정(NVIDIA가 고객에게서 연산 용량을 사는 계약)
    supply: float         # 공급·생산능력 약정(참고, D에 넣지 않음)
    dc_leases: float      # 미개시 데이터센터 리스(참고, D에 넣지 않음)
    revenue_q: float
    source: str


# 2026-07-26 이전 코멘터리는 클라우드 서비스 약정만 공시해 D를 계산할 수 없다.
# 2분기 보증 1,085억에는 분기 후(2026-08) 체결한 SB Energy 1,050억이 포함돼 있다(코멘터리 표 기준). 리스 45억 = 자체 25 + 제3자 재양도 예정 20.
COMMITMENTS = (
    Commitments("2026-07-26", 1085, 360, 250, 290, 2790, 450, 962, Q2FY27),
    Commitments("2026-10-25", 1180, 320, 200, 280, 3900, 740, 1158, Q3FY27),
)


# 분기 실적(8-K 2.02) 발표일. 점검표 C1이 켜진 날 계산에 쓴다(SEC EDGAR submissions 기준).
RELEASED = {"2026-07-26": "2026-08-26", "2026-10-25": "2026-11-17"}


# 데이터센터 매출의 고객 구분(억 달러). 2분기 코멘터리부터 고객 분류를 소급 변경했다(1분기 하이퍼스케일 378.7 → 430.5).
# 변경 후 기준으로 다시 공시된 분기만 둔다. 각 분기는 가장 최근 코멘터리의 값.
PLATFORM = (
    ("2025-07-27", 241.68, 169.28, Q2FY27),
    ("2025-10-26", 351, 161, Q3FY27),
    ("2026-04-26", 430.50, 321.96, Q2FY27),
    ("2026-07-26", 487.10, 403.13, Q2FY27),
    ("2026-10-25", 601, 475, Q3FY27),
)


def platform():
    """하이퍼스케일 vs AI 클라우드·산업·기업(네오클라우드 포함). 비중은 데이터센터 매출 중."""
    d = pd.DataFrame(PLATFORM, columns=["period_end", "hyperscale", "ai_clouds", "source"])
    d["period_end"] = pd.to_datetime(d.period_end)
    d["ai_clouds_share"] = d.ai_clouds / (d.hyperscale + d.ai_clouds)
    return d


def _quarters(sec):
    d = sec.loc[(sec.ticker == "NVDA") & sec.metric.isin(["revenue", "ar", "inventory", "ocf", "net_income"])]
    d = d.sort_values("filed").drop_duplicates(["end", "metric"], keep="first")
    return d.pivot(index="end", columns="metric", values="value").sort_index()


def _year_ago(q, end):
    """회계 분기가 13·14주라 정확히 1년 전 날짜가 없다. 1년 전 ±10일 안의 분기말."""
    near = q.index[abs((q.index - (end - pd.DateOffset(years=1))).days) <= 10]
    return near[-1] if len(near) else None


def table(sec):
    """분기별 A·B·C·D 값. 금액은 달러."""
    q = _quarters(sec).reindex(columns=["revenue", "ar", "inventory", "ocf", "net_income"])
    for c in COMMITMENTS:
        end = pd.Timestamp(c.period)
        if end not in q.index or pd.isna(q.at[end, "revenue"]):
            q.loc[end, "revenue"] = c.revenue_q * 1e8
    q = q.sort_index()
    days = q.index.to_series().diff().dt.days
    out = pd.DataFrame(index=q.index)
    out["revenue"] = q.revenue
    out["dso"] = q.ar / q.revenue * days
    consecutive = days.between(84, 105)
    window = consecutive.rolling(3).sum().eq(3)   # 직전 3개 간격이 모두 한 분기면 최근 4분기가 연속
    out["revenue_ttm"] = q.revenue.rolling(4).sum().where(window)
    ocf_ttm = q.ocf.rolling(4).sum().where(window)
    ni_ttm = q.net_income.rolling(4).sum().where(window)
    out["cash_conv"] = (ocf_ttm / ni_ttm).where(ni_ttm > 0)
    for end in q.index:
        prior = _year_ago(q, end)
        if prior is None:
            continue
        out.at[end, "dso_chg"] = out.at[end, "dso"] - out.at[prior, "dso"]
        out.at[end, "inv_gap"] = ((q.at[end, "inventory"] / q.at[prior, "inventory"] - 1)
                                  - (q.at[end, "revenue"] / q.at[prior, "revenue"] - 1))
    for c in COMMITMENTS:
        end = pd.Timestamp(c.period)
        total = (c.guarantees + c.neocloud + c.equity + c.cloud_service) * 1e8
        out.loc[end, ["support", "supply", "dc_leases"]] = total, c.supply * 1e8, c.dc_leases * 1e8
        out.loc[end, "support_ratio"] = total / out.at[end, "revenue_ttm"]
    return out.reindex(columns=["revenue", "revenue_ttm", "dso", "dso_chg", "inv_gap", "cash_conv",
                                "support", "support_ratio", "supply", "dc_leases"])


def _last(t, col):
    s = t[col].dropna()
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, None)


def flags(t):
    checks = []
    end, v = _last(t, "dso_chg")
    if end is not None:
        checks.append((v >= DSO_UP, f"A DSO {t.at[end, 'dso']:.0f}일(4분기 전 대비 {v:+.0f}일, {end:%Y-%m-%d})"))
    end, v = _last(t, "inv_gap")
    if end is not None:
        checks.append((v >= INV_GAP, f"B 재고 증가율 − 매출 증가율 {v:+.0%}p({end:%Y-%m-%d})"))
    end, v = _last(t, "cash_conv")
    if end is not None:
        checks.append((v < CASH_CONV, f"C 영업현금흐름 ÷ 순이익 {v:.2f}(최근 12개월, {end:%Y-%m-%d})"))
    ratio = t.support_ratio.dropna()
    if len(ratio) >= 2:
        chg = ratio.iloc[-1] - ratio.iloc[-2]
        checks.append((chg >= SUPPORT_UP, f"D 고객 지원성 약정 ÷ 매출 {ratio.iloc[-1]:.0%}(직전 분기 대비 {chg * 100:+.0f}%p, "
                                          f"{ratio.index[-1]:%Y-%m-%d})"))
    bad = sum(hit for hit, _ in checks)
    if len(checks) < 3:
        lvl = "보류"
    else:
        lvl = "경고" if bad >= 2 else "주의" if bad else "정상"
    text = " · ".join(f"{label}{' ▲악화' if hit else ''}" for hit, label in checks) or "계산할 수 있는 항목이 없음"
    return [{"id": "F16", "title": "수요의 질(NVIDIA 현금 회수·고객 금융)", "on": lvl in ("주의", "경고"), "level": lvl,
             "text": f"{text} — 악화 {bad}/{len(checks)}개. 고객 지원성 약정은 보증 최대 노출을 그대로 더한 값이라 현재 실제 노출보다 큽니다."}]
