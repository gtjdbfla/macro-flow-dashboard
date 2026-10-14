"""월가 자금: 지금 · 한국으로의 전달 · 과거 · 앞으로(확인 수준별)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from ui import badge, load, plot

S = load("results/summary.json")
sec = load("results/sector_snapshot.parquet")
ind = load("results/industry_snapshot.parquet")
cf = load("views/cftc_now.parquet")
kr_inv = load("views/kr_investor.parquet")
kr_sec = load("views/kr_sector_now.parquet")
quint = load("views/korea_quintile.parquet")
kh = load("views/korea_model_recent.parquet")
fh = load("views/flow_history.parquet")
meta = load("views/meta.json")
ev = load("views/evidence.json")

st.title("월가 자금")
st.markdown('<p class="lede">자금이 지금 어느 섹터에 있고, 과거에 어떻게 옮겨 다녔는지 봅니다. 앞날에 대해 통계로 확인된 것은 '
            '<b>미국 마감이 다음 한국 장에 미치는 영향</b>뿐이고, 나머지는 현황과 단서로만 표시합니다.</p>', unsafe_allow_html=True)
ui.freshness(meta)


def heat(z, x, y, lim, fmt="{:+.1f}", h=380):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim, showscale=False,
                               text=[[fmt.format(v) for v in row] for row in z], texttemplate="%{text}", xgap=2, ygap=2,
                               hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    plot(fig, h, legend=False)


# ── 지금 ────────────────────────────────────────────────────────────────────
st.header("1. 지금 — 자금은 어디에 있나")
order = sec.sort_values("flow_1m", ascending=False)
left, right = st.columns([2.1, 1], gap="large")
with left:
    a, b = st.columns(2)
    with a:
        st.markdown("**ETF 순유입** (순자산 대비 %, 실제 설정·환매)")
        heat((order[["flow_1w", "flow_1m", "flow_3m"]] * 100).values, ["1주", "1개월", "3개월"], list(order["name"]), 12)
    with b:
        st.markdown("**SPY 대비 상대강도** (%p, 가격)")
        heat((order[["rs_1w", "rs_1m", "rs_3m"]] * 100).values, ["1주", "1개월", "3개월"], list(order["name"]), 8)
    st.markdown('<p class="note">초록이 짙을수록 자금이 들어오거나 시장보다 강한 섹터입니다. 두 가지가 같은 방향이면 신호가 더 믿을 만하고, 엇갈리면 '
                '가격은 올랐지만 자금은 빠지는 식의 구간입니다. 섹터 자금 신호의 이후 수익 예측력은 아직 확인되지 않았습니다(아래 4절).</p>',
                unsafe_allow_html=True)
with right:
    st.markdown("**기관 선물 포지션** (자산운용사, 이력 대비 백분위)")
    c = cf.dropna(subset=["am_rank"]).sort_values("am_rank")
    fig = go.Figure(go.Bar(x=c.am_rank, y=c.name, orientation="h", marker_color=np.where(c.am_rank >= 50, "#22a55a", "#e08a1e"),
                           text=[f"{v:.0f}" for v in c.am_rank], textposition="outside", hovertemplate="%{y}: %{x:.0f}분위<extra></extra>"))
    fig.update_xaxes(range=[0, 110])
    plot(fig, 380, legend=False)
    st.markdown('<p class="note">S&P 섹터 E-mini 선물(CFTC 주간). 50보다 오른쪽이면 평소보다 순매수 쪽입니다. 화요일 기준·금요일 공표.</p>', unsafe_allow_html=True)
with st.expander("산업·테마 ETF (반도체·은행·바이오·방산 등)"):
    t = ind.assign(flow_1m=ind.flow_1m * 100, rs_1m=ind.rs_1m * 100, rs_1w=ind.rs_1w * 100)[["name", "score", "direction", "flow_1m", "rs_1m", "rs_1w", "flow_src"]]
    st.dataframe(t.rename(columns={"name": "산업", "score": "점수", "direction": "방향", "flow_1m": "1M 순유입%", "rs_1m": "1M RS%p", "rs_1w": "1W RS%p",
                                   "flow_src": "자금 출처"}).round(1), hide_index=True, width="stretch", height=320)

# ── 한국으로의 전달 ─────────────────────────────────────────────────────────
st.header("2. 한국으로의 전달 — 통계로 확인된 유일한 연결")
last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
z = float(last.pred_foreign_z)
k1, k2, k3 = st.columns([1.1, 1.2, 1.1], gap="large")
with k1:
    ev_k = ev["korea_eval"]["test"]
    st.markdown(badge("검증됨", "good") + badge(f"표본외 상관 {ev_k['corr']:.2f}", "mid"), unsafe_allow_html=True)
    st.metric("미국 마감 → 다음 한국 거래일 외국인 수급(z)", f"{z:+.2f}", "순매수 쪽" if z > 0.15 else ("순매도 쪽" if z < -0.15 else "중립"))
    q = quint[quint.period == "oos"]
    fig = go.Figure(go.Bar(x=q.q, y=q.buy_prob, marker_color="#5b8def", text=[f"{v:.0f}%" for v in q.buy_prob], textposition="outside"))
    fig.update_yaxes(range=[0, 80])
    plot(fig, 210, legend=False)
    st.markdown('<p class="note">2023년 이후 실제 결과. 예측값 5분위 중 가장 높은 구간은 외국인 순매수 확률이 약 60%, 가장 낮은 구간은 약 25%였습니다. '
                '설명력은 크지 않아 방향 힌트로만 쓰세요.</p>', unsafe_allow_html=True)
with k2:
    st.markdown("**KOSPI 투자자별 순매수 누적** (최근 120거래일, 조원)")
    d = kr_inv.tail(120)
    fig = go.Figure()
    for col, nm in [("frgn_ntby_tr_pbmn", "외국인"), ("orgn_ntby_tr_pbmn", "기관"), ("prsn_ntby_tr_pbmn", "개인")]:
        fig.add_scatter(x=d.index, y=(d[col] / 1e6).cumsum(), name=nm, mode="lines")
    plot(fig, 300)
    gp = quint[quint.period == "oos"]
    st.markdown(f'<p class="note"><b>중요</b>: 미국 신호가 맞히는 다음날 KOSPI 수익은 거의 시가 갭입니다(2023년 이후 상위−하위 5분위 차이 '
                f'{gp.total_bp.iloc[-1]-gp.total_bp.iloc[0]:.0f}bp 중 갭 {gp.gap_bp.iloc[-1]-gp.gap_bp.iloc[0]:.0f}bp). 개장 후 추가 초과수익은 확인되지 않았습니다.</p>',
                unsafe_allow_html=True)
with k3:
    st.markdown("**한국 업종별 외국인 순매수 비중** (20일, 거래대금 대비 %)")
    ks = kr_sec.sort_values("foreign_20d")
    fig = go.Figure(go.Bar(x=ks.foreign_20d, y=ks.sector, orientation="h", marker_color=np.where(ks.foreign_20d >= 0, "#22a55a", "#ee5555"),
                           text=[f"{v:+.1f}" for v in ks.foreign_20d], textposition="outside"))
    fig.update_xaxes(range=[min(-1, ks.foreign_20d.min() * 1.35), max(1, ks.foreign_20d.max() * 1.35)])
    plot(fig, 300, legend=False)
    st.markdown('<p class="note">업종 대표 종목 3~4개 합산(KIS). 현재 대형주로 고정돼 과거 구성과 다를 수 있습니다.</p>', unsafe_allow_html=True)


# ── 과거 ────────────────────────────────────────────────────────────────────
@st.fragment
def history():
    st.header("3. 과거 — 자금은 섹터 사이를 어떻게 옮겨 다녔나")
    per = st.segmented_control("기간", ["최근 1년", "3년", "5년", "2007~ 전체"], default="3년", label_visibility="collapsed", key="hist_per")
    n = {"최근 1년": 52, "3년": 156, "5년": 260, "2007~ 전체": len(fh)}[per or "3년"]
    d = fh.tail(n)
    secs = [c for c in d.columns if c != "regime"]
    codes = {"Goldilocks": 0, "Reflation": 1, "Slowdown": 2, "Stagflation": 3}
    colors = ["#22a55a", "#e0b41e", "#5b8def", "#ee5555"]
    rg = d["regime"].map(codes).astype(float)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.07, 0.93], vertical_spacing=0.015)
    fig.add_trace(go.Heatmap(z=[rg.values], x=d.index, colorscale=[[i / 3, c] for i, c in enumerate(colors)], zmin=0, zmax=3, showscale=False,
                             text=[d["regime"].fillna("").values], hovertemplate="%{x|%Y-%m-%d} %{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=(d[secs].T * 100).clip(-10, 10).values, x=d.index, y=secs, colorscale="RdYlGn", zmid=0, zmin=-10, zmax=10,
                             colorbar=dict(title="20일 순유입<br>%순자산", len=0.6, thickness=10),
                             hovertemplate="%{y} %{x|%Y-%m-%d}: %{z:+.1f}%<extra></extra>"), row=2, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    plot(fig, 420, legend=False)
    st.markdown('<p class="note">행은 섹터 ETF, 색은 최근 20일 순 설정유입(초록=유입, 빨강=유출)입니다. 위쪽 띠는 그 시점의 매크로 국면(초록 Goldilocks · 노랑 Reflation · '
                '파랑 Slowdown · 빨강 Stagflation). 국면별로 자금이 일정하게 같은 섹터로 갔다는 패턴은 통계로 확인되지 않았습니다(4절).</p>', unsafe_allow_html=True)


history()

# ── 앞으로 ──────────────────────────────────────────────────────────────────
st.header("4. 앞으로 — 확인된 수준별로 구분")
g1, g2, g3 = st.columns(3, gap="medium")
with g1:
    st.markdown(badge("검증됨", "good") + "**미국 마감 → 한국 다음 장**", unsafe_allow_html=True)
    st.markdown("2절의 연결입니다. 시차 약 1거래일, 세 개 구간(2005-18 · 2019-22 · 2023~) 모두 같은 방향이었습니다. 주로 시가 갭으로 나타납니다.")
with g2:
    st.markdown(badge("단서", "warn") + "**약하지만 일관된 신호**", unsafe_allow_html=True)
    st.markdown("- 한국 업종 외국인 60일 순매수 비중이 높을수록 이후 20일 업종 수익이 높은 경향(상관 약 0.07, 세 구간 모두 +)\n"
                "- 미국 섹터 ETF에 자금이 많이 들어온 뒤 이후 수익이 낮은 경향(역방향)\n"
                f"- 위 가설들은 앞으로 쌓이는 데이터로 다시 검증 중(전향 장부 {ev['ledger'].get('n_days', 0)}일째)")
with g3:
    st.markdown(badge("참고", "mid") + "**섹터·레짐 기반 예측은 근거 부족**", unsafe_allow_html=True)
    st.markdown(f"매크로 국면별 섹터 초과수익은 검정 {ev['regime_testable']:,}건 중 세 구간에서 모두 재현된 것이 {ev['regime_replicated']}건"
                f"({ev['regime_rate']*100:.1f}%)으로 우연 수준입니다. 섹터 상대강도·자금 신호의 예측력도 구간마다 부호가 바뀝니다.")


@st.fragment
def analog():
    A = S["analog"]
    st.subheader("비슷했던 과거 국면 뒤에는 (참고)")
    fw = load("results/analog_forward.parquet")
    assets = sorted(fw.asset.unique())
    c1, c2 = st.columns([1, 3])
    with c1:
        asset = st.selectbox("자산", assets, index=assets.index("SPY") if "SPY" in assets else 0, key="analog_asset")
        st.caption(f"유사 국면 {A['n_episodes']}사례({A['n_days']}일). 사례가 적으면 분포를 신뢰하지 마세요.")
    with c2:
        sub = fw[(fw.asset == asset) & (fw.kind == "abs")].sort_values("h")
        if len(sub):
            xs = [f"{h}일" for h in sub.h]
            fig = go.Figure()
            fig.add_bar(x=xs, y=(sub.p90 - sub.p10) * 100, base=sub.p10 * 100, name="유사 국면 10~90%", opacity=.35)
            fig.add_scatter(x=xs, y=sub["median"] * 100, mode="markers", name="유사 국면 중앙값", marker=dict(size=11, color="#f59e0b"))
            fig.add_scatter(x=xs, y=sub.base_median * 100, mode="markers", name="평소 중앙값", marker=dict(size=9, symbol="diamond", color="#8a99a3"))
            fig.update_yaxes(title="이후 수익률 (%)")
            plot(fig, 260)


analog()
