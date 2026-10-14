"""① 월가 자금: 과거(자금은 어떻게 움직였나) → 현재(지금 어디에 있나) → 다음(확인된 수준별)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from ui import badge, ev_card, kpis, load, plot, trio

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
state = load("views/state.json")

last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
z = float(last.pred_foreign_z)
zdir = "순매수 쪽" if z > 0.15 else ("순매도 쪽" if z < -0.15 else "중립")

st.title("① 월가 자금")
ui.purpose("월가의 돈은 어디에서 어디로 움직였고, 지금 어디에 있고, 다음에 어디로 갈까?",
           f"지금 매크로 국면은 <b>{state['regime']}</b>({state['days_in_regime']}거래일째)입니다. 통계로 확인된 앞날 정보는 "
           f"<b>미국 마감 → 다음 한국 장 외국인 수급</b> 하나뿐이고(현재 <b>{zdir}</b>), 섹터 간 자금 이동을 미리 맞히는 신호는 확인되지 않았습니다.",
           [("과거", "섹터 ETF 자금이 시기별·국면별로 어디로 갔나 (2007~)"),
            ("현재", "지금 어느 섹터에 자금이 들어오고 빠지나 · 기관 선물 포지션"),
            ("다음", "미국 → 한국 전달(검증됨) · 약한 단서 · 근거 부족한 것")],
           badge("한국 전달만 검증됨", "info") + badge("섹터 이동은 약한 단서", "warn"))
ui.legend()
ui.freshness(meta)


def heat(zv, x, y, lim, fmt="{:+.1f}", h=380):
    fig = go.Figure(go.Heatmap(z=zv, x=x, y=y, colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim, showscale=False,
                               text=[[fmt.format(v) for v in row] for row in zv], texttemplate="%{text}", textfont=dict(size=14), xgap=2, ygap=2,
                               hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=13))
    fig.update_xaxes(side="top")
    plot(fig, h, legend=False)


# ── 과거 ────────────────────────────────────────────────────────────────────
@st.fragment
def history():
    ui.stage("과거", "자금은 섹터 사이를 어떻게 옮겨 다녔나", "기간을 골라 보세요")
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
    plot(fig, 440, legend=False)
    st.markdown('<p class="note">행은 섹터 ETF, 색은 최근 20일 순 설정유입(초록=유입, 빨강=유출)입니다. 위쪽 띠는 그 시점의 매크로 국면 '
                '(초록 Goldilocks · 노랑 Reflation · 파랑 Slowdown · 빨강 Stagflation). '
                '국면별로 자금이 일정하게 같은 섹터로 갔다는 패턴은 통계로 확인되지 않았습니다(아래 "다음" 절).</p>', unsafe_allow_html=True)


history()

# ── 현재 ────────────────────────────────────────────────────────────────────
ui.stage("현재", "자금은 지금 어디에 있나", "미국 섹터 ETF · 기관 선물")
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
                '가격은 올랐지만 자금은 빠지는 식의 구간입니다. 섹터 자금 신호의 이후 수익 예측력은 아직 확인되지 않았습니다.</p>',
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

# ── 다음 ────────────────────────────────────────────────────────────────────
ui.stage("다음", "한국으로의 전달 — 통계로 확인된 유일한 연결")
q = quint[quint.period == "oos"]
ev_k = ev["korea_eval"]["test"]
st.markdown(badge("검증됨", "good") + badge("표본외 2023~ 기준", "mid"), unsafe_allow_html=True)
st.markdown(kpis([("다음 한국 거래일 외국인 수급 (z)", f"{z:+.2f}", f"미국 마감 기준 · {zdir}"),
                  ("모델 표본외 상관", f"{ev_k['corr']:.2f}", f"설명력(R²) {ev_k['r2_vs_mean']:.2f} · 방향 힌트 수준"),
                  ("예측 최상위 vs 최하위 구간", f"{q.buy_prob.iloc[-1]:.0f}% vs {q.buy_prob.iloc[0]:.0f}%", "외국인 순매수 확률 (평소 약 47%)")]), unsafe_allow_html=True)
k1, k2, k3 = st.columns(3, gap="large")
with k1:
    st.markdown("**예측 5분위별 외국인 순매수 확률 (%)**")
    fig = go.Figure(go.Bar(x=["Q1 약세", "Q2", "Q3", "Q4", "Q5 강세"], y=q.buy_prob, marker_color="#5b8def", text=[f"{v:.0f}%" for v in q.buy_prob], textposition="outside"))
    fig.update_yaxes(range=[0, 80])
    plot(fig, 320, legend=False)
    st.markdown('<p class="note">2023년 이후 실제 결과. 예측값이 높은 구간일수록 순매수 확률이 높습니다.</p>', unsafe_allow_html=True)
with k2:
    st.markdown("**KOSPI 투자자별 순매수 누적 (최근 120거래일, 조원)**")
    d = kr_inv.tail(120)
    fig = go.Figure()
    for col, nm in [("frgn_ntby_tr_pbmn", "외국인"), ("orgn_ntby_tr_pbmn", "기관"), ("prsn_ntby_tr_pbmn", "개인")]:
        fig.add_scatter(x=d.index, y=(d[col] / 1e6).cumsum(), name=nm, mode="lines")
    plot(fig, 320)
    gp = quint[quint.period == "oos"]
    st.markdown(f'<p class="note">미국 신호가 맞히는 다음날 KOSPI 수익은 거의 시가 갭입니다(상위−하위 {gp.total_bp.iloc[-1]-gp.total_bp.iloc[0]:.0f}bp 중 갭 '
                f'{gp.gap_bp.iloc[-1]-gp.gap_bp.iloc[0]:.0f}bp). 개장 후 추가 초과수익은 확인되지 않았습니다.</p>', unsafe_allow_html=True)
with k3:
    st.markdown("**한국 업종별 외국인 순매수 비중 (20일, 거래대금 대비 %)**")
    ks = kr_sec.sort_values("foreign_20d")
    fig = go.Figure(go.Bar(x=ks.foreign_20d, y=ks.sector, orientation="h", marker_color=np.where(ks.foreign_20d >= 0, "#22a55a", "#ee5555"),
                           text=[f"{v:+.1f}" for v in ks.foreign_20d], textposition="outside"))
    fig.update_xaxes(range=[min(-1, ks.foreign_20d.min() * 1.4), max(1, ks.foreign_20d.max() * 1.4)])
    plot(fig, 320, legend=False)
    st.markdown('<p class="note">업종 대표 종목 3~4개 합산(KIS). 대표 종목은 현재 대형주로 고정돼 과거 구성과 다를 수 있습니다.</p>', unsafe_allow_html=True)

st.markdown("#### 확인된 수준별 정리")
kv = ev["korea_eval"]
trio(
    ev_card(badge("검증됨", "good"), "미국 마감 → 한국 다음 장", "시차 약 1일", "세 구간 모두 같은 방향, 주로 시가 갭으로 나타남",
            [("2005–18 상관", f"{kv['train']['corr']:.2f}", ""), ("2019–22 상관", f"{kv['val']['corr']:.2f}", ""), ("2023~ 표본외 상관", f"{kv['test']['corr']:.2f}", "")]),
    ev_card(badge("단서", "warn"), "약하지만 방향이 일관된 신호", "상관 약 0.07", "한국 업종 외국인 60일 흐름 → 이후 20일 업종 수익 (세 구간 모두 +)",
            [("미국 ETF 순유입 → 이후 상대수익", "역방향", "약함"), ("VIX가 클수록 이후 수익", "높은 경향", "가설 반대"),
             ("전향 장부", f"{ev['ledger'].get('n_days', 0)}일째", "재검증 중")]),
    ev_card(badge("참고", "mid"), "섹터·국면 기반 예측은 근거 부족", f"{ev['regime_rate']*100:.1f}%", f"세 구간에서 모두 재현된 비율 (검정 {ev['regime_testable']:,}건 중 {ev['regime_replicated']}건, 우연 기대 ≈1.25%)",
            [("섹터 상대강도·자금 신호", "구간마다 부호 변동", ""), ("점수 구성요소 채택", f"{ev['score_included']}개", f"/ {ev['score_components']}개")]))


@st.fragment
def analog():
    A = S["analog"]
    st.markdown("#### 비슷했던 과거 국면 뒤에는 (참고)")
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
