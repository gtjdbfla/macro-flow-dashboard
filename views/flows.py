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
hd = load("views/sector_holdings.parquet")

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

_T = load("views/evidence_tests.parquet")
_P = _T[_T.primary].set_index("claim")
_k = ev["korea_eval"]["test"]
_q = quint[quint.period == "oos"]
ui.howto(
    info_md=(
        "**과거 — 자금이 어디로 옮겨 다녔나**\n"
        "- 미국 섹터 ETF 11개의 **실제 설정·환매**(발행좌수 변화 × NAV)로 만든 20일 순유입(%순자산), 2007년~. 가격이 아니라 돈의 이동입니다.\n"
        "- 그 시점의 **매크로 국면**(성장·물가 → Goldilocks/Reflation/Slowdown/Stagflation, 21일 평활 + 이력 보정으로 연 4회 안팎만 전환).\n\n"
        "**현재 — 지금 어디에 있나**\n"
        "- 섹터별 20일 순유입 **금액**과 **비율**, SPY 대비 상대강도, 기관(자산운용사) **선물 포지션**(CFTC 주간), 섹터 대표 종목과 ETF 내 비중(SPDR 공개 파일).\n"
        "- 한국: 외국인·기관·개인 **투자자별 순매수**와 8개 업종 외국인 순매수 비중(한국투자증권 API, 2001~).\n\n"
        "**알 수 없는 것**: iShares·VanEck 등 다른 운용사 ETF의 자금, **스타일 팩터(성장·가치·퀄리티)의 실제 자금**, 옵션 포지셔닝, 13F. 무료 데이터가 없습니다.\n\n"
        "<span class='muted'>갱신: 미국은 평일 마감 후, 한국은 평일 장 마감 후, CFTC는 금요일, 팩터·산업 데이터는 월 1회.</span>"
    ),
    pred_md=(
        "이 탭은 **\"어디로 갈지 맞힌다\"고 주장하지 않습니다.** 과거 패턴이 세 구간(2005–18·2019–22·2023~)에서 같은 방향으로 반복됐는지를 규칙으로 걸러, "
        "통과한 것만 등급과 함께 보여줍니다.\n\n"
        + badge("검증됨", "good") + f" **미국 마감 → 다음 한국 거래일 외국인 수급.** 입력은 QQQ·반도체(SOXX)·달러·실질금리 변화·직전 한국 외국인 수급 5개를 선형 회귀 하나로 묶은 모델입니다. "
        f"표본외(2023~) 상관 {_k['corr']:.2f}, 설명력 {_k['r2_vs_mean']*100:.0f}%로 방향 힌트 수준이며, 예측 최상위 구간의 순매수 확률은 " + f"{_q.buy_prob.iloc[-1]:.0f}%, 최하위는 {_q.buy_prob.iloc[0]:.0f}%" + "(평소 약 47%)입니다. 효과는 대부분 시가 갭에서 나옵니다.\n\n"
        + badge("검증됨", "good") + f" **팩터 모멘텀(가격).** 최근 12개월 오른 팩터(시장·소형·가치·수익성·투자)는 다음 달에도 오르는 경향. 1963년~, 전체 t {_P.loc['factor_momentum'].all_t:+.1f}, 월 평균 약 +0.3%로 효과는 작습니다. "
        "그래서 \"다음에 자산이 몰릴 팩터\"는 **최근 12개월 강한 쪽**이 답이며, 자금이 아니라 가격 추세 근거입니다.\n\n"
        + badge("검증됨", "good") + f" **산업 모멘텀.** 1927~1999년 t 6이지만 2000년 이후 t {_P.loc['industry_momentum'].test_t:.1f}로 약해졌습니다(알려진 뒤 약화).\n\n"
        + badge("단서", "warn") + f" **자금이 몰린 섹터는 이후 20일 덜 오르는 경향.** 11섹터를 매일 비교한 상관, 전체 t {_P.loc['us_flow_contrarian'].all_t:+.1f}. 2019–22만 반대여서 확정하지 못합니다.\n\n"
        + badge("참고", "mid") + " **국면 → 섹터 수익, 한국 외국인 순매수 지속, 섹터 가격 모멘텀(ETF 시대)**은 우연 수준이라 예측에 쓰지 않습니다. 국면 규칙은 검정 " + f"{ev['regime_testable']:,}건 중 세 구간 재현이 {ev['regime_rate']*100:.1f}%" + "로 우연 기대(약 1.25%)와 같은 수준이었습니다.\n\n"
        "**신뢰성 장치**: 구간 경계의 미래 윈도우 제거(purge), 겹치는 수익률은 Newey-West로 보정, 호라이즌은 사전에 하나만 지정, "
        "그리고 모든 후보를 **전향 장부 H1~H7**에 사전 등록해 앞으로 쌓이는 실제 데이터로 다시 판정합니다."
    ),
)


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
if hd is not None:
    st.markdown("#### 섹터별 대표 종목")
    st.markdown(ui.holdings_table(hd, list(order["name"])), unsafe_allow_html=True)
    st.markdown(f'<p class="note">섹터 ETF(SPDR) 공개 보유 종목 기준 {hd.as_of.iloc[0]:%Y-%m-%d}. 위 히트맵과 같은 순서(1개월 순유입 순)입니다. '
                '시가총액 비중이라 소수 대형주가 섹터 움직임을 좌우합니다.</p>', unsafe_allow_html=True)
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
st.markdown('<p class="note">입력: 미국 QQQ·SOXX·달러지수·실질금리 + 직전 한국 거래일 외국인 수급(모델 v2, 2026-10-14). 직전 수급을 더해 표본외 상관이 0.33에서 0.45로 올랐습니다(근거·방법 페이지).</p>', unsafe_allow_html=True)
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

st.markdown("#### 다음에 몰릴 곳 — 확인된 수준별 정리")
T = load("views/evidence_tests.parquet")
en = load("views/evidence_now.json")
st.markdown(ui.grade_table(T), unsafe_allow_html=True)
st.markdown('<p class="note">구간: 미국 ETF·한국 종목은 2005–18 / 2019–22 / 2023~, 산업 모멘텀은 1927–62 / 1963–99 / 2000~, 팩터 모멘텀은 1963–89 / 1990–2009 / 2010~. '
            '<b>검증됨</b> = 세 구간 같은 부호 + 2구간 이상 |t|≥2 + 전체 |t|≥3, <b>단서</b> = 전체 |t|≥2 + 2구간 이상 같은 부호, 나머지 <b>참고</b>. '
            '호라이즌이 여럿인 주장은 사전에 정한 대표 하나만 판정에 씁니다. 이 평가는 이미 본 과거 데이터에 대한 것이고, 전향 장부에서 다시 확인합니다.</p>',
            unsafe_allow_html=True)
f1, f2 = st.columns(2, gap="large")
with f1:
    st.markdown(f"**팩터 12개월 누적 수익** ({en['asof'][:7]} 기준 · 팩터 모멘텀의 현재 신호)")
    fv = pd.Series(en["factor_12m"]).sort_values()
    fig = go.Figure(go.Bar(x=fv.values * 100, y=[ui.FACTOR_KO[k] for k in fv.index], orientation="h", marker_color=np.where(fv.values >= 0, "#22a55a", "#ee5555"),
                           text=[f"{v*100:+.0f}%" for v in fv.values], textposition="outside"))
    fig.update_xaxes(range=[min(-5, fv.min() * 100 * 1.5), max(5, fv.max() * 100 * 1.5)])
    plot(fig, 300, legend=False)
    st.markdown('<p class="note">12개월 누적이 +인 팩터는 다음 달에도 +일 경향(월 평균 +0.2~0.3%p). 가치·소형·수익성·투자 팩터는 롱숏 포트폴리오 수익입니다. '
                '효과가 작고 월별로 들쭉날쭉해, 방향 힌트로만 쓰세요. 데이터는 지난달까지입니다(발표 지연).</p>', unsafe_allow_html=True)
with f2:
    st.markdown(f"**산업 12−1개월 모멘텀 상위·하위 5** ({en['asof'][:7]} 기준, 49개 산업)")
    top = [(ui.IND_KO.get(k.strip(), k), v) for k, v in en["industry_top"]]
    bot = [(ui.IND_KO.get(k.strip(), k), v) for k, v in en["industry_bottom"]]
    rws = lambda L: "".join(f"<tr><td>{n}</td><td class='r'>{v*100:+.0f}%</td></tr>" for n, v in L)  # noqa: E731
    st.markdown(f"<table class='mini'><tr><th>많이 오른 산업</th><th class='r'>11개월 수익</th></tr>{rws(top)}</table><div style='height:10px'></div>"
                f"<table class='mini'><tr><th>많이 못 오른 산업</th><th class='r'>11개월 수익</th></tr>{rws(bot)}</table>", unsafe_allow_html=True)
    st.markdown('<p class="note">1927~1999년에는 강한 규칙(t 6)이었으나 2000년 이후는 약해졌습니다(t 1.4, 같은 방향). 미국 섹터 ETF(2005~)에서는 확인되지 않았습니다.</p>', unsafe_allow_html=True)


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
