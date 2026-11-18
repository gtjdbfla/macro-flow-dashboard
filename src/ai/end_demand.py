"""최종 수요 — AI에 돈을 내는가. F17 Ramp AI 도입률, F18 소프트웨어 바스켓 매출화. 규칙은 docs/AI_PROFITABILITY.md에 사전 등록."""
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
    """회사별 최신 분기의 매출 전년비, 4분기 전 전년비, 가속도, 매출총이익률과 전년 대비 변화."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    d = sec.loc[sec.ticker.isin(SOFTWARE) & sec.metric.isin(["revenue", "cost_rev", "gross_profit"]) & (sec.filed <= as_of)]
    d = d.sort_values("filed").drop_duplicates(["ticker", "end", "metric"], keep="first")
    rows, hist = [], []
    for ticker in SOFTWARE:
        q = d[d.ticker == ticker].pivot(index="end", columns="metric", values="value")
        q = q.reindex(columns=["revenue", "cost_rev", "gross_profit"]).sort_index()
        q = q[q.revenue > 0]
        if q.empty:
            rows.append({"ticker": ticker, "name": NAMES[ticker]})
            continue
        gm = (q.gross_profit / q.revenue).fillna(1 - q.cost_rev / q.revenue)
        yoy = pd.Series({end: q.at[end, "revenue"] / q.at[prior, "revenue"] - 1
                         for end in q.index if (prior := _year_ago(q.index, end)) is not None}, dtype=float)
        hist += [{"ticker": ticker, "quarter_end": end, "yoy": v} for end, v in yoy.items()]
        end = q.index[-1]
        row = {"ticker": ticker, "name": NAMES[ticker], "quarter_end": end, "recent": (as_of - end).days <= RECENT_DAYS,
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
