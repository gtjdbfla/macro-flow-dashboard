"""핵심 현황: 매크로 환경과 AI 수익성 두 가지만, 지금 판단에 필요한 것만 한 화면에 모은다. 나머지는 '상세' 메뉴에 그대로 있다."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import risk_view
import ui
from src.ai import circular, end_demand
from ui import load, plot

meta = load("views/meta.json")
st.title("핵심 현황")
st.caption("두 가지 질문만 봅니다. 시장 환경은 어떤가, 그리고 AI 투자는 수익으로 돌아오고 있는가. 근거·검증·자금 흐름은 위 '상세' 메뉴에 있습니다.")
try:
    ui.freshness(meta)
except Exception:
    pass

# ── 1. 매크로 ───────────────────────────────────────────────────────────────
st.markdown("### 1. 매크로와 시장 환경")
try:
    risk_view.render(meta)
except Exception as e:
    st.caption(f"매크로 카드를 표시하지 못했습니다: {type(e).__name__}")

ew = load("results/ai_early_warning.json")
if ew:
    sg = ew.get("stage", {})
    hot = [i for i in ew["indicators"] if i["level"] in ("red", "yellow")]
    st.markdown(f"**시장 단계: {sg.get('name', '-')}** — {sg.get('why', '')}")
    if hot:
        st.dataframe(pd.DataFrame([{"": "🔴" if i["level"] == "red" else "🟡", "경고 지표": i["name"], "상태": i["detail"]} for i in hot]),
                     hide_index=True, width="stretch")
    ui.note("조기경보는 하락이 시작된 뒤에 확인되는 후행 지표이고 예측이 아닙니다. 과거 기록상 신호가 켜졌을 때 이후 하락 비율은 기준률보다 조금 높은 수준이었습니다(전체 표는 상세 > AI 인프라).")

# 직접 관리하는 일정(날짜가 바뀌면 손으로 고친다)
EVENTS = (
    ("2026-12-09", "FOMC 결정", "시장은 25bp 인상을 거의 100% 반영. 이후 경로(3월 추가 인상 여부)가 쟁점"),
    ("2026-12 중순", "11월 CPI", "근원 물가 월 0.3% 이상이 계속되는지"),
    ("2026-12 초", "NVIDIA 3분기 10-Q", "지리별 매출(중국·싱가포르), 매출채권 회수 기간, 현금 전환율"),
    ("2026-12 중순", "Oracle 분기 실적", "RPO, CAPEX, 차입"),
    ("2026-12-31", "Fermi 착공 통지 기한", "보유 종목 개별 사건"),
    ("2027-01 말~02 초", "빅테크 4사 4분기 실적", "2027년 CAPEX 가이던스, 영업현금흐름, 잉여현금흐름. 가장 중요한 분기점"),
    ("2027-02 말", "NVIDIA 4분기 실적", "고객 지원성 약정 비율, 가이던스 달성"),
)
with st.expander("앞으로 확인할 일정", expanded=True):
    st.dataframe(pd.DataFrame(EVENTS, columns=["시점", "사건", "볼 것"]), hide_index=True, width="stretch")

# ── 2. AI 수익성 ────────────────────────────────────────────────────────────
st.markdown("### 2. AI 수익성")
latest = load("results/ai_latest.parquet")
hist = load("results/ai_history.parquet")
cloud = load("results/ai_cloud_latest.parquet")
guide = load("results/ai_guide.json")
dq = load("results/ai_demand_quality.parquet")

tiles = []
if latest is not None and (latest.ticker == "BIG4").any():
    B = latest[latest.ticker == "BIG4"].iloc[0]
    tiles += [("CAPEX 증가 vs 매출 증가", f"{B.capex_yoy:+.0%} / {B.rev_yoy:+.0%}", "빅테크 4사, 최근 12개월 전년비"),
              ("CAPEX ÷ 영업현금흐름", f"{B.capex_ocf:.0%}", "번 돈 중 설비에 쓰는 비율"),
              ("잉여현금흐름(12개월)", f"{B.fcf_ttm / 1e8:,.0f}억 달러", f"직전 분기 대비 {B.fcf_chg / 1e8:+,.0f}억")]
if cloud is not None:
    c = cloud.set_index("ticker")
    parts = [f"{n} {c.loc[t, 'yoy']:+.0%}" for t, n in (("AMZN", "AWS"), ("GOOGL", "Google Cloud")) if t in c.index and pd.notna(c.loc[t, "yoy"])]
    if parts:
        tiles.append(("클라우드 매출 성장(전년비)", " · ".join(parts), "최신 분기"))
aq = end_demand.anthropic_quarters()
if len(aq):
    tiles.append(("Anthropic 컴퓨팅 비용 ÷ 매출", f"{aq.compute_ratio.iloc[-1]:.0%}", f"{aq.quarter_end.iloc[-1]:%Y-%m} 분기, 매출 {aq.revenue.iloc[-1] / 100:,.0f}억 달러"))
if dq is not None and dq.support_ratio.notna().any():
    s = dq.dropna(subset=["support_ratio"]).iloc[-1]
    tiles.append(("NVIDIA 고객 지원성 약정 ÷ 매출", f"{s.support_ratio:.0%}", f"{pd.to_datetime(s.period_end):%Y-%m} 분기 (12개월 매출 기준)"))
if tiles:
    st.markdown(ui.kpis(tiles), unsafe_allow_html=True)

if guide and guide.get("snapshot", {}).get("bubble"):
    bb = guide["snapshot"]["bubble"]
    mark = {True: "예", False: "아니오", None: "판정 불가"}
    n_ok = sum(i["ok"] is True for i in bb["resolution"])
    n_w = sum(i["ok"] is True for i in bb["warning"])
    st.markdown(f"**AI 수익성 의문 점검표** — 해소 조건 {n_ok}/{len(bb['resolution'])} 충족 · 악화 신호 {n_w}/{len(bb['warning'])} 발생")
    ui.bubble_tables(bb, {"A": "현금 회수", "B": "최종 수요", "C": "금융", "D": "시장 안정"}, side_by_side=False, res_height=590, warn_height=215)
    ui.note("시장이 의심하는 것(번 돈보다 쓰는 돈이 크다, 순환 자금, 높은 금리)이 풀렸는지를 매주 같은 틀로 보는 목록입니다. 지금 대부분 '아니오'인 것은 기준을 현재 값을 알고 정했기 때문입니다. 합계에 예측력이 있다고 주장하지 않습니다.")

c1, c2 = st.columns(2, gap="large")
with c1:
    if hist is not None:
        st.markdown("**번 돈 vs 쓰는 돈** (4사 12개월 합, 억 달러)")
        b = hist[hist.ticker == "BIG4"].dropna(subset=["capex_ttm", "ocf_ttm"]).sort_values("period_end")
        b = b[b.period_end >= "2018-01-01"]
        fig = go.Figure()
        fig.add_scatter(x=b.period_end, y=b.ocf_ttm / 1e8, name="영업현금흐름", mode="lines", line=dict(color="#22a55a", width=2.6))
        fig.add_scatter(x=b.period_end, y=b.capex_ttm / 1e8, name="CAPEX", mode="lines", line=dict(color="#f97316", width=2.6),
                        fill="tonexty", fillcolor="rgba(238,85,85,0.18)")
        plot(fig, 280, key="core_capex")
with c2:
    st.markdown("**Anthropic 매출과 컴퓨팅 비용** (억 달러)")
    fig = go.Figure()
    fig.add_bar(x=aq.quarter_end, y=aq.revenue / 100, name="매출", marker_color="#5b8def")
    fig.add_bar(x=aq.quarter_end, y=aq.compute / 100, name="컴퓨팅·인프라 비용", marker_color="#f97316")
    fig.update_layout(barmode="group")
    plot(fig, 280, key="core_anthropic")

with st.expander("순환 구조 — 공시로 확인한 돈의 연결고리", expanded=False):
    ft = circular.facts()
    st.dataframe(ft, hide_index=True, width="stretch", column_config={"출처": st.column_config.LinkColumn("원문", display_text="열기")})
    ui.note("공급자·클라우드가 AI 기업에 투자하면 그 돈이 다시 그 회사의 칩과 클라우드를 사는 데 쓰입니다. 회사 공시 값만 옮겼고 판정은 하지 않습니다. 단위는 억 달러입니다.")

hp = load("results/ai_holdings_prices.parquet")
if hp is not None:
    st.markdown("### 3. 내 보유 종목")
    pt = pd.DataFrame({"보유": np.where(hp.owned, "★", ""), "종목": hp.name, "고점 대비%": hp.dd_52w * 100, "1개월%": hp.ret_1m * 100, "3개월%": hp.ret_3m * 100})
    st.dataframe(pt[pt.보유 == "★"].sort_values("고점 대비%").round(1), hide_index=True, width="stretch")
    ui.note(f"기준일 {pd.to_datetime(hp.last_date).max():%Y-%m-%d}, 수정주가 종가. 보유 종목은 모두 같은 AI 투자 수요에 묶여 있어 같은 신호에 함께 노출됩니다. 업종 비교와 회사별 실적은 상세 > AI 인프라에 있습니다.")
