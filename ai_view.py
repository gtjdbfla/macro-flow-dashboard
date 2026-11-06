"""AI 인프라 수요, 현금 회수, 조달 여력의 첫 화면."""
import pandas as pd
import streamlit as st

import ui
from src.ai import guidance

# XBRL 공통 태그로 확보하지 못하는 주석은 공시 기준일과 원문을 함께 표시한다.
CRWV_DATE = "2026-06-30"
CRWV_FILING = "https://www.sec.gov/Archives/edgar/data/1769628/000176962826000366/crwv-20260630.htm"
CRWV_AVAILABLE_CREDIT = 10.014e9
FILING_NOTES = (
    ("CoreWeave", CRWV_DATE, "수주잔고 1,037억 달러 중 41%를 첫 24개월 내 매출로 인식할 것으로 예상. "
     "분기 매출의 62%는 상위 두 고객에서 발생.", CRWV_FILING),
    ("Microsoft", "2026-09-30", "분기 중 금융리스로 취득한 사용권자산 52억 달러. 현금 CAPEX와 취득 방식이 다름.",
     "https://www.sec.gov/Archives/edgar/data/789019/000119312526436133/msft-20260930.htm"),
    ("Alphabet", "2026-09-30", "구매약정 9,240억 달러와 미개시 리스 및 기타 의무 1,520억 달러를 공시. "
     "일부 약정은 가격·물량에 따라 실제 지출이 달라질 수 있음.",
     "https://abc.xyz/investor/events/event-details/default.aspx"),
    ("Meta", "2026-09-30", "미개시 리스 3,557억 달러. 데이터센터·에너지·네트워크 인프라 등을 포함.",
     "https://www.sec.gov/Archives/edgar/data/1326801/000162828026066773/meta-20260930.htm"),
)


def demand_tiles(cloud):
    if cloud is None or cloud.empty:
        return []
    current = cloud.sort_values("quarter_end").drop_duplicates("ticker", keep="last").set_index("ticker")
    tiles = []
    for ticker, name in (("AMZN", "AWS"), ("GOOGL", "Google Cloud"), ("MSFT", "Azure")):
        if ticker not in current.index:
            tiles.append((name, "—", "자료 없음"))
            continue
        row = current.loc[ticker]
        growth = row.azure_yoy if ticker == "MSFT" else row.yoy
        period = pd.Timestamp(row.quarter_end).strftime("%Y-%m")
        margin = f" · 영업이익률 {row.margin:.1%}" if ticker != "MSFT" and pd.notna(row.margin) else ""
        tiles.append((name, f"{growth:+.1%}" if pd.notna(growth) else "—",
                      f"{period} 분기 · 매출 전년비{margin}"))
    return tiles


def _funding_table(latest, safety):
    if safety is None or safety.empty:
        st.caption("Oracle·CoreWeave의 재무 자료를 확인 중입니다.")
        return
    balance = safety.set_index("ticker")
    rows = []
    for ticker in ("ORCL", "CRWV"):
        if ticker not in latest.index or ticker not in balance.index:
            continue
        fin, debt = latest.loc[ticker], balance.loc[ticker]
        period = pd.Timestamp(fin.period_end).date().isoformat()
        cash_date = debt.get("cash_asof")
        limit = CRWV_AVAILABLE_CREDIT if ticker == "CRWV" and period == cash_date == CRWV_DATE else None
        compatible = period == cash_date and pd.notna(debt.need12) and debt.need12 > 0
        rows.append({
            "회사": fin["name"], "실적 기준일": period, "현금 기준일": cash_date or "—",
            "현금·단기투자": debt.cash / 1e8, "미사용 약정 한도": limit / 1e8 if limit is not None else None,
            "현금만 (년)": debt.cash / debt.need12 if compatible else None,
            "한도 포함 (년)": (debt.cash + limit) / debt.need12 if compatible and limit is not None else None,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "현금·단기투자": st.column_config.NumberColumn(format="%.1f"),
            "미사용 약정 한도": st.column_config.NumberColumn(format="%.1f"),
            "현금만 (년)": st.column_config.NumberColumn(format="%.2f"),
            "한도 포함 (년)": st.column_config.NumberColumn(format="%.2f"),
        })
        st.caption("금액은 억 달러. 기간은 최근 12개월 FCF 적자와 12개월 내 원금 만기를 합친 금액을 기준으로 단순 환산했습니다. "
                   "약정 한도는 인출 조건이 있는 차입 여력입니다. 빈칸은 같은 기준일의 확인 자료가 없는 항목입니다.")
        if any(row["미사용 약정 한도"] is not None for row in rows):
            st.markdown(f"[CoreWeave 약정 한도 원문]({CRWV_FILING})")


def _guidance_tracker():
    st.markdown("#### 회사가 제시한 숫자와 실제 실적")
    rows = guidance.comparisons(ui.load_parquet_dir("sec_quarterly.parquet"))
    observed = rows["실제"].notna()
    reported, pending, backlog = st.tabs(["발표된 실적", "확인 대기", "수주잔고의 매출 인식"])
    links = {
        "제시 원문": st.column_config.LinkColumn(display_text="보기"),
        "실적 원문": st.column_config.LinkColumn(display_text="보기"),
        "같은 분기 회사 전체 FCF (억 달러)": st.column_config.NumberColumn(format="%+.1f"),
    }
    with reported:
        st.dataframe(rows[observed], hide_index=True, width="stretch", column_config=links)
        st.caption("같은 회사·분기·지표·환율 기준으로 비교합니다. 회사 전체 FCF는 해당 분기의 영업현금흐름에서 "
                   "현금 CAPEX를 뺀 값이며, 사업부 매출 지표 옆에서는 회사 전체의 현금 상황을 보여줍니다.")
        st.markdown(
            f"**AI 매출에 가까운 직접 공개치** · Microsoft는 9월 분기 Foundry 매출이 전년의 3배를 넘었다고 밝혔고, "
            f"Alphabet은 연초부터 9월까지 Google Cloud의 AI 매출이 수백억 달러 규모라고 밝혔습니다. "
            f"[Microsoft 실적 설명]({guidance.MSFT_OCT}) · [Alphabet 실적 설명]({guidance.GOOGL_OCT})"
        )
    with pending:
        upcoming = rows[~observed].drop(columns=["실제", "실적 원문"])
        if len(upcoming):
            st.dataframe(upcoming.dropna(axis=1, how="all"), hide_index=True, width="stretch", column_config=links)
        else:
            st.caption("등록된 항목의 실적을 모두 확인했습니다.")
        st.caption("비교할 실제 수치가 들어오면 ‘발표된 실적’으로 이동합니다. 회사 전체 매출·영업이익은 SEC 자료와 "
                   "연결하며, 개별 클라우드 수치와 회사 가이던스는 원문을 확인해 등록합니다.")
    with backlog:
        rows = [{
            "회사": name, "기준일": day, "공시 범위": scope, "잔고 (억 달러)": total,
            "인식 기간": f"향후 {months}개월", "인식 예정 비중": share,
            "인식 예정액 환산 (억 달러)": round(total * ratio), "공시 원문": source,
        } for name, day, scope, total, months, share, ratio, source in guidance.BACKLOG]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "공시 원문": st.column_config.LinkColumn(display_text="보기"),
        })
        st.caption("예정액은 회사가 발표한 비중으로 환산한 근삿값입니다. Alphabet은 55%를 적용했습니다. "
                   "기간과 계약 범위가 회사마다 다르며, 인식 예정 매출과 현금 수취 시점은 다를 수 있습니다.")


def fundamentals(latest, cloud, safety):
    latest = latest.set_index("ticker")
    big = latest.loc["BIG4"]
    period = pd.Timestamp(big.period_end).strftime("%Y-%m")
    st.subheader("1. 수요가 매출과 이익으로 이어지나")
    tiles = demand_tiles(cloud)
    if tiles:
        st.markdown(ui.kpis(tiles), unsafe_allow_html=True)
    st.caption("Azure는 회사가 발표한 매출 성장률입니다. AWS·Google Cloud의 부문 마진과 Azure의 매출 성장률은 각각의 공시 범위로 읽습니다.")

    _guidance_tracker()

    st.subheader("2. 인프라 투자비를 빼면 현금이 얼마나 남나")
    st.markdown(ui.kpis([
        ("Big4 영업현금흐름", f"{big.ocf_ttm / 1e8:,.0f}억 달러", "최근 12개월"),
        ("현금 CAPEX", f"{big.capex_ttm / 1e8:,.0f}억 달러", f"영업현금흐름의 {big.capex_ocf:.0%}"),
        ("잉여현금흐름", f"{big.fcf_ttm / 1e8:,.0f}억 달러", f"전년 대비 {big.fcf_chg / 1e8:+,.0f}억 달러"),
    ]), unsafe_allow_html=True)
    cols = ["name", "period_end", "ocf_ttm", "capex_ttm", "fcf_ttm", "fcf_chg"]
    bridge = latest.loc[latest.index.intersection(["BIG4", "MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRWV"]), cols].copy()
    bridge["period_end"] = bridge.period_end.dt.strftime("%Y-%m-%d")
    bridge.iloc[:, 2:] /= 1e8
    bridge.columns = ["회사", "12개월 종료일", "영업현금흐름", "현금 CAPEX", "잉여현금흐름", "FCF 전년 대비"]
    with st.expander("회사별 현금흐름 보기"):
        st.dataframe(bridge.round(1), hide_index=True, width="stretch")
    st.caption(f"Big4는 {period}까지 최근 12개월 회사 전체 합계입니다. 단위는 억 달러. FCF는 영업현금흐름에서 "
               "현금 CAPEX를 차감한 공통 산식입니다. 리스 취득과 장기 약정은 아래 공시 주석에서 따로 확인할 수 있습니다.")

    if pd.notna(big.capex_ocf) and big.ocf_ttm > 0:
        required_growth = big.capex_ocf * 0.20
        st.markdown(f"**현금 회수 기준점** · Big4의 현금 CAPEX가 20% 늘어난다고 가정하면, "
                    f"현재 FCF를 유지하려면 영업현금흐름이 약 {required_growth:.1%} 늘어야 합니다.")

    st.subheader("3. 회수될 때까지 자금을 댈 수 있나")
    _funding_table(latest, safety)
    with st.expander("공시 주석 · 수주 인식, 고객 집중, 리스·구매약정"):
        for name, day, note, link in FILING_NOTES:
            st.markdown(f"- **{name} · {day}** {note} [원문]({link})")
        st.caption("위 주석은 표시된 분기의 원문에서 발췌한 참고 기록입니다. 차기 분기의 자동 수집 지표와 갱신 주기가 다릅니다.")
    st.markdown("**다음 실적에서 확인할 것** · 클라우드 성장과 마진이 유지되는지, 영업현금흐름 증가액이 "
                "CAPEX 증가액에 가까워지는지, 수주잔고가 실제 매출과 현금 유입으로 전환되는지.")
