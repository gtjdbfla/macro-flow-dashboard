"""AI 수익성: 투자 → 클라우드 수요 → 이익·현금 → 조달, 4단계 체인으로 본다."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import badge, level_kind, load, load_parquet_dir, pct, plot

meta = load("views/meta.json")
latest = load("results/ai_latest.parquet")
hist = load("results/ai_history.parquet")
cloud_h = load("results/ai_cloud_history.parquet")
agg = load("results/ai_cloud_agg.parquet")
spreads = load("results/ai_credit_spreads.parquet")
issu = load("results/ai_credit_issuance.parquet")
tw = load_parquet_dir("twse_monthly.parquet")
disc = load_parquet_dir("ai_disclosures.parquet")
fobj = json.loads(open(ui.DATA_DIR / "results/ai_flags.json", encoding="utf-8").read())
fl = {f["id"]: f for f in fobj["flags"]}
B = latest[latest.ticker == "BIG4"].iloc[0]
LT = latest.set_index("ticker")
cl = agg.iloc[-1]

st.title("AI 수익성")
st.markdown('<p class="lede">AI 투자가 수익으로 돌아오고 있는지를 <b>투자 → 클라우드 수요 → 이익·현금 → 조달</b> 네 단계로 나눠 봅니다. '
            '공시 재무는 회사 전체 기준이라 AI만의 수치는 아닙니다. 이 페이지는 현재 상태와 방향을 보여줄 뿐 주가를 예측하지 않습니다.</p>', unsafe_allow_html=True)
ui.freshness(meta)


def kind_of(ids, bad_if_any=False):
    ks = [level_kind(fl[i]["level"]) for i in ids if i in fl]
    if not ks:
        return "mid"
    if any(k == "bad" for k in ks):
        return "bad"
    if any(k == "warn" for k in ks):
        return "warn"
    return "good" if all(k == "good" for k in ks) else "mid"


n_tw = 0
if tw is not None and len(tw):
    m = tw[tw.ticker.isin(["2330", "2317", "2382", "6669", "3231", "2356", "2376", "2345"])].sort_values("month").groupby("ticker").tail(1)
    n_tw = int((m.yoy > 20).sum())
g10 = spreads[(spreads.ticker == "GOOGL") & (spreads.bucket == "10Y") & (spreads.currency == "USD") & (spreads.filed >= "2025-01-01")].sort_values("filed")
g10_txt = f"{g10.spread_bp.iloc[0]:.0f}→{g10.spread_bp.iloc[-1]:.0f}bp" if len(g10) >= 2 else "—"
steps = [
    ("1 투자", "투자(CAPEX)", "warn" if B.capex_yoy > 0.5 else "mid",
     f"Big4 12개월 CAPEX <b>{B.capex_ttm/1e8:,.0f}억 달러</b><br>전년 대비 <b>{pct(B.capex_yoy*100)}</b>"),
    ("2 수요", "클라우드·공급망", kind_of(["F4", "F6", "F7", "F8"]),
     f"클라우드 3사 매출 <b>{pct(cl.yoy*100)}</b> · 마진 {cl.margin*100:.0f}%<br>대만 핵심 8사 중 <b>{n_tw}사</b> 월매출 +20%↑"),
    ("3 이익·현금", "회수와 현금흐름", kind_of(["F1", "F2"]),
     f"매출 {pct(B.rev_yoy*100)} · 영업이익 {pct(B.opinc_yoy*100)}<br>FCF 전년 대비 <b>{B.fcf_chg/1e8:+,.0f}억 달러</b>"),
    ("4 조달", "자금 조달", kind_of(["F3", "F9", "F10"]),
     f"오라클 CAPEX/영업현금흐름 <b>{LT.loc['ORCL','capex_ocf']:.2f}</b> · 코어위브 <b>{LT.loc['CRWV','capex_ocf']:.1f}</b><br>알파벳 10년 신규발행 가산금리 {g10_txt}"),
]
lab = {"good": "양호", "warn": "주의", "bad": "경고", "mid": "중립"}
html = '<div class="chain">' + "".join(
    f'<div class="step {k}"><div class="n">STEP {n}</div><div class="t">{t} {badge(lab[k], k)}</div><div class="kv">{kv}</div></div>' for n, t, k, kv in steps) + "</div>"
st.markdown(html, unsafe_allow_html=True)
st.markdown('<p class="note">상태 판정은 사전에 고정한 규칙의 신호 10개를 단계별로 묶은 것입니다(근거·방법 페이지). 종합 점수는 가중치를 검증할 표본이 없어 만들지 않았습니다.</p>',
            unsafe_allow_html=True)

# ── 변화 ────────────────────────────────────────────────────────────────────
st.header("1. 어떻게 변해 왔나")
b4 = hist[hist.ticker == "BIG4"].dropna(subset=["capex_yoy"]).sort_values("period_end").tail(12)
a, b = st.columns(2, gap="large")
with a:
    st.markdown("**투자는 매출·이익보다 빠르다** (Big4, 최근 12개월 기준 전년 대비 %)")
    fig = go.Figure()
    for c, n, col in [("capex_yoy", "CAPEX", "#f97316"), ("rev_yoy", "매출", "#5b8def"), ("opinc_yoy", "영업이익", "#22a55a")]:
        fig.add_scatter(x=b4.period_end, y=b4[c] * 100, name=n, mode="lines+markers", line=dict(color=col, width=2.4))
    plot(fig, 300)
with b:
    st.markdown("**클라우드는 가속하고 마진도 오른다** (AWS·Google Cloud·Microsoft IC 합계)")
    ca = agg.tail(10)
    fig = go.Figure()
    fig.add_bar(x=ca.quarter_end, y=ca.yoy * 100, name="매출 증가율(%)", marker_color="#5b8def", opacity=.8)
    fig.add_scatter(x=ca.quarter_end, y=ca.margin * 100, name="영업이익률(%)", mode="lines+markers", line=dict(color="#f59e0b", width=2.4))
    plot(fig, 300)

c, d = st.columns([1.25, 1], gap="large")
with c:
    st.markdown("**회사별 현황**")
    order = ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRWV", "NVDA", "BIG4"]
    t = latest.set_index("ticker").loc[order].reset_index()
    loss = t.op_margin < 0
    tab = pd.DataFrame({"회사": t["name"], "CAPEX(억$)": t.capex_ttm / 1e8, "CAPEX 증가%": t.capex_yoy * 100, "매출 증가%": t.rev_yoy * 100,
                        "영업이익 증가%": np.where(loss, np.nan, t.opinc_yoy * 100), "FCF(억$)": t.fcf_ttm / 1e8,
                        "CAPEX/OCF": np.where(t.ticker == "NVDA", np.nan, t.capex_ocf)})
    st.dataframe(tab.round(1), hide_index=True, width="stretch", height=330,
                 column_config={"CAPEX(억$)": st.column_config.NumberColumn(format="%.0f"), "FCF(억$)": st.column_config.NumberColumn(format="%.0f")})
    st.markdown('<p class="note">NVIDIA는 투자를 받는 쪽이라 CAPEX/OCF를 비웠고, 적자 기업(코어위브)의 영업이익 증가율은 비웠습니다.</p>', unsafe_allow_html=True)
with d:
    @st.fragment
    def spread_chart():
        st.markdown("**빅테크 신규 채권 가산금리** (벤치마크 대비 bp)")
        bk = st.segmented_control("만기", ["10Y", "5Y", "30Y"], default="10Y", label_visibility="collapsed", key="sp_bk")
        sq = spreads[(spreads.bucket == (bk or "10Y")) & (spreads.filed >= "2023-01-01") & (spreads.currency == "USD")]
        fig = go.Figure()
        for tk, g in sq.groupby("ticker"):
            fig.add_scatter(x=g.filed, y=g.spread_bp, name=tk, mode="lines+markers")
        plot(fig, 270)
    spread_chart()
    crwv = issu[issu.ticker == "CRWV"].sort_values("filed")
    if len(crwv):
        st.markdown('<p class="note">코어위브(하이일드) 신규 채권 쿠폰: ' + ", ".join(f"{r.coupon:.2f}% {r.currency}" for r in crwv.itertuples()) + "</p>", unsafe_allow_html=True)

# ── 앞으로 ──────────────────────────────────────────────────────────────────
st.header("2. 앞으로 — 예측이 아니라 두 가지 도구")
st.markdown('<p class="note">이 지표들은 주가를 예측한 적이 없습니다(이벤트 스터디에서 CAPEX 가속·감속이 반도체 수익을 구분하지 못함). '
            '대신 가정을 직접 넣어 결과를 보는 산식 시나리오와, 다음에 확인될 선행 지표를 제공합니다.</p>', unsafe_allow_html=True)
e1, e2 = st.columns(2, gap="large")
with e1:
    @st.fragment
    def scenario():
        st.markdown(badge("산식 시나리오", "info") + "**12개월 뒤 Big4의 CAPEX와 영업현금흐름**", unsafe_allow_html=True)
        cur_c = float(np.clip(B.capex_yoy * 100, -20, 120))
        ocf_y = hist[hist.ticker == "BIG4"].dropna(subset=["ocf_yoy"]).sort_values("period_end").iloc[-1].ocf_yoy * 100
        cg = st.slider("CAPEX 증가율 (%)", -20, 120, int(round(cur_c)), key="sc_c")
        og = st.slider("영업현금흐름 증가율 (%)", -20, 80, int(round(float(np.clip(ocf_y, -20, 80)))), key="sc_o")
        cx, ox = B.capex_ttm * (1 + cg / 100), B.ocf_ttm * (1 + og / 100)
        m1, m2, m3 = st.columns(3)
        m1.metric("CAPEX", f"{cx/1e8:,.0f}억 달러", f"{(cx-B.capex_ttm)/1e8:+,.0f}")
        m2.metric("영업현금흐름", f"{ox/1e8:,.0f}억 달러", f"{(ox-B.ocf_ttm)/1e8:+,.0f}")
        m3.metric("잉여현금흐름", f"{(ox-cx)/1e8:,.0f}억 달러", f"{((ox-cx)-B.fcf_ttm)/1e8:+,.0f}")
        ratio = cx / ox
        st.markdown(f"CAPEX/영업현금흐름 <b>{B.capex_ocf*100:.0f}% → {ratio*100:.0f}%</b>. "
                    + ("영업현금흐름보다 투자가 커져 부채·증자 의존이 늘어납니다." if ratio > 1 else
                       "영업현금흐름으로 감당 가능한 범위입니다." if ratio < 0.85 else "여유가 거의 없는 구간입니다."), unsafe_allow_html=True)
        st.markdown('<p class="note">슬라이더 기본값은 최근 12개월의 실제 증가율입니다. 단순 곱셈이라 투자 시점 지연·감가상각·자사주 매입은 반영하지 않습니다.</p>', unsafe_allow_html=True)
    scenario()
with e2:
    st.markdown(badge("선행 지표", "info") + "**대만 AI 공급망 월매출** (전년 동월 대비 %)", unsafe_allow_html=True)
    if tw is not None and len(tw):
        lm = tw.sort_values("month").groupby("ticker").tail(1).sort_values("yoy")
        fig = go.Figure(go.Bar(x=lm.yoy, y=lm["name"].str.split("(").str[0], orientation="h", marker_color=np.where(lm.yoy >= 20, "#22a55a", "#e08a1e"),
                               text=[f"{v:+.0f}%" for v in lm.yoy], textposition="outside"))
        fig.update_xaxes(range=[0, max(130, lm.yoy.max() * 1.15)])
        plot(fig, 300, legend=False)
        st.markdown(f'<p class="note">{lm.month.max().strftime("%Y-%m")}월 기준. 하이퍼스케일러 투자가 공급사 매출로 가는 속도를 분기 공시보다 1~2개월 먼저 보여줍니다. '
                    f'이력은 수집을 시작한 시점부터 쌓입니다. 다음 확인: 매월 10일경, 분기 공시는 10월 말.</p>', unsafe_allow_html=True)

with st.expander("경영진이 직접 공개한 AI 수치 (원문, 신뢰 등급 A)"):
    dd = disc.sort_values("filed", ascending=False).head(10)
    st.dataframe(pd.DataFrame({"회사": dd.ticker, "분기": dd.quarter_end.dt.strftime("%Y-%m"), "원문": dd.text}), hide_index=True, width="stretch")
with st.expander("신호 10개의 규칙과 현재 판정"):
    for i in sorted(fl, key=lambda x: int(x[1:])):
        f = fl[i]
        st.markdown(badge(f["level"], level_kind(f["level"])) + f"**{i} {f['title']}** — {f['text']}", unsafe_allow_html=True)
