"""최종 수요 — AI에 돈을 내는가. F17 Ramp AI 도입률, F18 소프트웨어 바스켓 매출화. 규칙은 docs/AI_PROFITABILITY.md에 사전 등록."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.data.sec import SOFTWARE

RAMP = {"RAMPAIALL": "전체", "RAMPAISMB": "소기업", "RAMPAIMIDMARKET": "중견기업", "RAMPAIENTERPRISE": "대기업"}
ADOPT_UP, MAX_AGE = 0.03, 75
ACCEL, GM_DROP, MIN_COMPANIES, RECENT_DAYS = 0.02, -0.02, 7, 183
NAMES = {"CRM": "Salesforce", "NOW": "ServiceNow", "ADBE": "Adobe", "INTU": "Intuit", "PLTR": "Palantir",
         "WDAY": "Workday", "SNOW": "Snowflake", "DDOG": "Datadog", "MDB": "MongoDB", "HUBS": "HubSpot"}


def ramp(as_of=None):
    """월별 도입률(0~1). as_of에 볼 수 있던 최신 공표본(vintage) 하나만 쓴다.
    Ramp가 2026-10-14 방법을 바꾸며 과거를 소급 수정해(소기업 8월 62.5% → 49.5%) 관측월마다 최초 값을 고르면 기준이 섞인다."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    out = {}
    for sid, name in RAMP.items():
        path = PARQUET / f"fredv_{sid}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        d = d[d.available_at <= as_of].sort_values("available_at").drop_duplicates("date", keep="last")
        out[name] = d.set_index("date")["value"].sort_index() / 100
    return pd.DataFrame(out)


def _year_ago(index, end):
    near = index[abs((index - (end - pd.DateOffset(years=1))).days) <= 10]
    return near[-1] if len(near) else None


def software(sec, as_of=None):
    return basket(sec, SOFTWARE, NAMES, as_of)


def basket(sec, tickers, names, as_of=None):
    """회사별 최신 분기의 매출 전년비, 4분기 전 전년비, 가속도, 매출총이익률과 전년 대비 변화."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    d = sec.loc[sec.ticker.isin(tickers) & sec.metric.isin(["revenue", "cost_rev", "gross_profit"]) & (sec.filed <= as_of)]
    d = d.sort_values("filed").drop_duplicates(["ticker", "end", "metric"], keep="first")
    rows, hist = [], []
    for ticker in tickers:
        q = d[d.ticker == ticker].pivot(index="end", columns="metric", values="value")
        q = q.reindex(columns=["revenue", "cost_rev", "gross_profit"]).sort_index()
        q = q[q.revenue > 0]
        if q.empty:
            rows.append({"ticker": ticker, "name": names[ticker]})
            continue
        gm = (q.gross_profit / q.revenue).fillna(1 - q.cost_rev / q.revenue)
        yoy = pd.Series({end: q.at[end, "revenue"] / q.at[prior, "revenue"] - 1
                         for end in q.index if (prior := _year_ago(q.index, end)) is not None}, dtype=float)
        hist += [{"ticker": ticker, "quarter_end": end, "yoy": v} for end, v in yoy.items()]
        end = q.index[-1]
        row = {"ticker": ticker, "name": names[ticker], "quarter_end": end, "recent": (as_of - end).days <= RECENT_DAYS,
               "yoy": yoy.get(end), "gm": gm.get(end)}
        prior = _year_ago(q.index, end)
        if prior is not None:
            row["yoy_prev"] = yoy.get(prior)
            row["gm_chg"] = gm.get(end) - gm.get(prior)
        row["accel"] = row["yoy"] - row["yoy_prev"] if pd.notna(row.get("yoy")) and pd.notna(row.get("yoy_prev")) else None
        rows.append(row)
    t = pd.DataFrame(rows).reindex(columns=["ticker", "name", "quarter_end", "recent", "yoy", "yoy_prev", "accel", "gm", "gm_chg"])
    return t, pd.DataFrame(hist, columns=["ticker", "quarter_end", "yoy"])


def flags(r, t, as_of=None):
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    out = []
    s = r["전체"].dropna() if "전체" in r else pd.Series(dtype=float)
    if s.empty or (as_of - s.index[-1]).days > MAX_AGE:
        out.append({"id": "F17", "title": "AI 도입 확산(Ramp)", "on": False, "level": "보류",
                    "text": "최신 Ramp AI Index 관측이 없거나 75일보다 오래돼 판정하지 않음."})
    else:
        last = s.index[-1]
        base = s.get(last - pd.DateOffset(months=6))
        if base is None or pd.isna(base):
            out.append({"id": "F17", "title": "AI 도입 확산(Ramp)", "on": False, "level": "보류", "text": "6개월 전 관측이 없음."})
        else:
            chg = s.iloc[-1] - base
            lvl = "긍정" if chg >= ADOPT_UP else "중립" if chg >= 0 else "주의"
            out.append({"id": "F17", "title": "AI 도입 확산(Ramp)", "on": lvl == "주의", "level": lvl,
                        "text": f"{last:%Y-%m} 미국 기업(Ramp 고객) {s.iloc[-1]:.1%}가 AI에 지출 — 6개월 전 대비 {chg * 100:+.1f}%p. "
                                "지출 여부만 재고 금액은 아님."})
    # 중앙값은 각 값이 있는 회사로 따로 낸다(Intuit·Workday는 표준 매출원가 태그가 없어 매출총이익률이 비어 있다).
    ok = t[(t.recent == True) & t.accel.notna()]  # noqa: E712
    ok_gm = t[(t.recent == True) & t.gm_chg.notna()]  # noqa: E712
    if len(ok) < MIN_COMPANIES:
        out.append({"id": "F18", "title": "소프트웨어 AI 매출화(10곳 바스켓)", "on": False, "level": "보류",
                    "text": f"최근 분기 비교가 가능한 회사 {len(ok)}곳 — {MIN_COMPANIES}곳 미만이라 판정하지 않음."})
        return out
    acc, gm = ok.accel.median(), ok_gm.gm_chg.median() if len(ok_gm) else 0.0
    if gm <= GM_DROP:
        lvl = "주의"
    else:
        lvl = "긍정" if acc >= ACCEL else "둔화" if acc <= -ACCEL else "중립"
    up = ", ".join(f"{r.name} {r.accel * 100:+.0f}%p" for r in ok.sort_values("accel", ascending=False).head(3).itertuples())
    out.append({"id": "F18", "title": "소프트웨어 AI 매출화(10곳 바스켓)", "on": lvl in ("주의", "둔화"), "level": lvl,
                "text": f"{len(ok)}곳 중앙값: 매출 전년비 {ok.yoy.median():+.0%}(1년 전 {ok.yoy_prev.median():+.0%}, 가속도 {acc * 100:+.1f}%p), "
                        f"매출총이익률 전년 대비 {gm * 100:+.1f}%p({len(ok_gm)}곳). 가속 상위: {up}. AI 이외의 요인(경기·가격·인수)도 섞입니다."})
    return out


# 회사가 실적 발표(8-K Ex.99.1)에서 밝힌 AI 연환산 매출. 금액 단위 억 달러. 판정 없이 표시만 한다(docs/AI_PROFITABILITY.md).
# Microsoft는 2026-09 분기 부문 개편 뒤 AI 연환산 매출을 내지 않았다. Amazon은 칩 사업(Trainium 등)과 따로 밝힌 AI 사업만 쓴다.
AI_REVENUE = (
    ("MSFT", "2024-12-31", 130, "AI 사업 연환산 매출 130억 달러 초과(전년 대비 +175%)",
     "https://www.sec.gov/Archives/edgar/data/789019/000095017025010484/msft-ex99_1.htm"),
    ("MSFT", "2026-03-31", 370, "AI 사업 연환산 매출 370억 달러 초과(전년 대비 +123%)",
     "https://www.sec.gov/Archives/edgar/data/789019/000119312526191457/msft-ex99_1.htm"),
    ("AMZN", "2026-06-30", 250, "AI 사업 연환산 매출 250억 달러 초과(칩 사업과 별도)",
     "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000024/amzn-20260630xex991.htm"),
)
# Gemini 1차 모델 API 처리량(분당 10억 토큰). 발표 문구가 "over"·"more than"·"nearly"라 근사값이다.
GEMINI_TOKENS = (
    ("2025-09-30", 7, "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000087/googexhibit991q32025.htm"),
    ("2025-12-31", 10, "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000012/googexhibit991q42025.htm"),
    ("2026-03-31", 16, "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000043/googexhibit991q12026.htm"),
    ("2026-06-30", 22, "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000066/googexhibit991q22026.htm"),
    ("2026-09-30", 32, "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000080/googexhibit991q32026.htm"),
)


def ai_revenue_coverage(hist):
    """AI 연환산 매출 ÷ 같은 분기말까지 12개월 현금 CAPEX. hist는 ai_history(ticker, period_end, capex_ttm)."""
    rows = []
    for ticker, end, run_rate, text, source in AI_REVENUE:
        h = hist[(hist.ticker == ticker) & (hist.period_end == pd.Timestamp(end))]
        capex = float(h.capex_ttm.iloc[0]) / 1e8 if len(h) and pd.notna(h.capex_ttm.iloc[0]) else None
        rows.append({"ticker": ticker, "period_end": pd.Timestamp(end), "run_rate": run_rate, "capex_ttm": capex,
                     "coverage": run_rate / capex if capex else None, "text": text, "source": source})
    return pd.DataFrame(rows)


def _priced(df):
    d = df[~df.id.str.contains(":") & (df.completion > 0) & (df.prompt >= 0)]
    return d


def token_price_index(df):
    """같은 모델의 게시가격 변화만 이어 붙인 지수(첫 수집일 = 100). 신모델 출시가 아니라 기존 모델의 가격 인하를 잰다."""
    d = _priced(df)
    days = sorted(d.fetched.unique())
    level, out = 100.0, []
    for i, day in enumerate(days):
        if i:
            a = d[d.fetched == days[i - 1]].set_index("id").completion
            b = d[d.fetched == day].set_index("id").completion
            both = a.index.intersection(b.index)
            if len(both):
                level *= float(np.exp(np.log(b[both] / a[both]).mean()))
        out.append({"fetched": day, "index": level, "models": int((d.fetched == day).sum())})
    return pd.DataFrame(out)


def new_model_prices(df):
    """최신 수집본에서 출시 월별 모델의 출력 가격 중앙값(USD/100만 토큰). 지금도 목록에 남은 모델만이라 생존 편향이 있다."""
    d = _priced(df)
    if d.empty:
        return pd.DataFrame(columns=["month", "median_completion", "models"])
    d = d[d.fetched == d.fetched.max()]
    g = d.groupby(d.created.dt.to_period("M"))
    out = pd.DataFrame({"median_completion": g.completion.median(), "models": g.size()}).reset_index(names="month")
    out["month"] = out.month.dt.to_timestamp()
    return out[out.month >= "2024-01-01"]


# Anthropic(최종 투자설명서 424B4, 2026-11-19 제출; S-1/A 3차와 분기 숫자 동일) 분기 실적. 금액 단위 백만 달러. 비상장 AI 연구소의 첫 1차 재무 공시라 판정 없이 표시만 한다.
# 영업손익은 GAAP이고 주식 기반 보상 등이 들어간다. 상장 전 자료라 수정본에서 숫자가 바뀔 수 있다.
ANTHROPIC_S1 = "https://www.sec.gov/Archives/edgar/data/2130357/000162828026074944/anthropicpbc-424b4.htm"
ANTHROPIC_QUARTERS = (
    ("2025-03-31", 376, 944, -935), ("2025-06-30", 787, 1390, -1200), ("2025-09-30", 1382, 2047, -3089),
    ("2025-12-31", 2040, 2945, -2836), ("2026-03-31", 4731, 3387, -1667), ("2026-06-30", 11555, 5558, 974),
    ("2026-09-30", 16956, 9863, -567),
)
# 억 달러. 영업현금흐름은 2026-01~09 누적, 약정은 구매약정·리스(미개시 포함)로 거의 전부 컴퓨팅·인프라.
ANTHROPIC_FACTS = (
    ("2026-01~09 영업현금흐름", 66, "작년 같은 기간 -20억"),
    ("2026-09-30 현금·단기투자", 1318.45, "상장 전 기준"),
    ("컴퓨팅 등 약정 합계", 9009, "미개시 리스 포함"),
    ("ㄴ 2030년 말까지 지출 예정", 4516, ""),
    ("ㄴ 앞으로 12개월 안 지출 예정", 707, ""),
)


def anthropic_quarters():
    d = pd.DataFrame(ANTHROPIC_QUARTERS, columns=["quarter_end", "revenue", "compute", "op_income"])
    d["quarter_end"] = pd.to_datetime(d.quarter_end)
    d["compute_ratio"] = d.compute / d.revenue
    d["op_margin"] = d.op_income / d.revenue
    return d


# CoreWeave 유통시장 신용 스트레스(언론 보도, 일자별). F10은 신규 발행 쿠폰만 봐서 유통 수익률과 CDS를 반영하지 못한다.
# CDS는 Deutsche Bank 인용(CNBC). 일자는 보도가 가리키는 거래일이다.
CRWV_STRESS = (
    ("2026-11-05", 1289, "5년 CDS(bp), 10월 고점 상회", "https://www.cnbc.com/2026/11/06/coreweave-bond-yield-tops-16percent-as-ai-credit-stress-rises.html"),
    ("2026-11-06", 1155, "5년 CDS(bp), 전일 대비 -8%", "https://www.cnbc.com/2026/11/06/coreweave-bond-yield-tops-16percent-as-ai-credit-stress-rises.html"),
    ("2026-11-06", 16, "2032년 만기 9.625% 채권 유통 수익률(%) 초과", "https://www.cnbc.com/2026/11/06/coreweave-bond-yield-tops-16percent-as-ai-credit-stress-rises.html"),
)
