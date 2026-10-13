"""Macro & Capital Flow Dashboard.
Macro → Liquidity → Wall Street Positioning → Sector Rotation → Korea Transmission.
모든 숫자는 data/results의 백테스트·검증 산출물에서 읽는다. 검증되지 않은 점수는 그렇게 표시한다."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import DATA_DIR, PARQUET

R = DATA_DIR / "results"
st.set_page_config(page_title="Macro & Capital Flow", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.74rem;font-weight:600}
.ok{background:#16653422;color:#16a34a;border:1px solid #16a34a55}
.no{background:#b4530922;color:#d97706;border:1px solid #d9770655}
.chip{display:inline-block;margin:2px 6px 2px 0;padding:3px 11px;border-radius:14px;background:#64748b22;
 border:1px solid #64748b55;font-size:.9rem}
.small{font-size:.8rem;color:#94a3b8}
div[data-testid="stMetricValue"]{font-size:2.1rem}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def rd(name):
    p = R / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=300)
def feat(name):
    return pd.read_parquet(PARQUET / f"features_{name}.parquet")


@st.cache_data(ttl=300)
def summary():
    p = R / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


S = summary()
st.title("Macro & Capital Flow")
if S is None:
    st.warning("분석 결과가 아직 없습니다 (scripts/run_scoring.py).")
    st.stop()
st.caption(f"기준일 {S['as_of']} (미국 마감 기준) · 펀더멘털 제외, 외부 시장환경만 · 점수 근거는 아래 '검증' 표기를 확인")
state = feat("state").dropna(subset=["state_key"]).iloc[-1]


def arrow(label, up, down):
    return "↑" if label == up else ("↓" if label == down else "→")


# ───────────────────────── 0. MONEY FLOW BRIEFING ─────────────────────────
st.header("MONEY FLOW BRIEFING")
for it in S.get("briefing", []):
    tag = '<span class="badge ok">검증</span>' if it["grade"] == "검증" else '<span class="badge no">참고용</span>'
    st.markdown(f"**{it['q']}** {tag}<br>{it['text']}", unsafe_allow_html=True)
st.caption("위 문장은 계산값을 그대로 옮긴 요약입니다. '참고용'은 예측력이 확인되지 않은 현재 위치 설명입니다.")
st.divider()

# ───────────────────────── 1. GLOBAL MARKET REGIME ─────────────────────────
st.header("GLOBAL MARKET REGIME")
cards = [("Macro", "macro"), ("Liquidity", "liquidity"), ("Risk Appetite", "risk"),
         ("Institutional Flow", "flow"), ("Korea Transmission", "korea_score")]
cols = st.columns(5)
for col, (title, key) in zip(cols, cards):
    v = S["scores"].get(key)
    validated = S["validated"].get(f"{key}_validated") if key != "korea_score" else True
    with col:
        st.metric(title, "—" if v is None else f"{v:.0f}/100")
        if key == "korea_score":
            ev = S["korea_model"]["eval"]
            st.markdown('<span class="badge ok">검증됨</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="small">표본외(2023~) 상관 {ev["test"]["corr"]:.2f}</span>', unsafe_allow_html=True)
        else:
            ic = S["oos_ic"][key]["test"]["ic"]
            tag = '<span class="badge ok">검증됨</span>' if validated else '<span class="badge no">미검증 · 동일가중 요약</span>'
            st.markdown(tag, unsafe_allow_html=True)
            if ic is not None:
                st.markdown(f'<span class="small">표본외(2023~) IC {ic:+.2f}</span>', unsafe_allow_html=True)

chips = [f"Regime: {state['regime']}", f"Growth {arrow(state['growth'], 'Accelerating', 'Decelerating')}",
         f"Inflation {arrow(state['inflation'], 'Accelerating', 'Decelerating')}", f"Policy: {state['policy']}",
         f"Liquidity: {state['liquidity']}", f"USD: {state['dollar']}"]
st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)
st.markdown('<div class="small">미검증 = 구성요소 중 train+val에서 SPY 20일 선행수익을 안정적으로 예측한 것이 하나도 없어 '
            '가중치를 줄 근거가 없다는 뜻입니다. 현재 위치를 보는 기술적 요약이지 예측이 아닙니다.</div>', unsafe_allow_html=True)

hist = rd("scores_history.parquet")
if hist is not None:
    h = hist.loc["2021-01-01":, ["macro", "liquidity", "risk", "flow"]].dropna(how="all")
    fig = go.Figure()
    names = {"macro": "Macro", "liquidity": "Liquidity", "risk": "Risk Appetite", "flow": "Institutional Flow"}
    for c in h.columns:
        fig.add_scatter(x=h.index, y=h[c].rolling(5).mean(), name=names[c], mode="lines", line=dict(width=1.6))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=8, b=0), legend=dict(orientation="h", y=1.18),
                      yaxis=dict(range=[0, 100], title="점수 (5일 평균)"))
    st.plotly_chart(fig, width="stretch")

with st.expander("점수 구성요소와 가중치 검증 (왜 이 점수인가)"):
    V = rd("score_validation.parquet")
    if V is not None:
        now = S["components_now"]
        V = V.copy()
        V["현재(0~1)"] = [now.get(f"{a}|{b}") for a, b in zip(V.score, V.component)]
        V = V.rename(columns={"score": "점수", "component": "구성요소", "ic_train": "IC train", "ic_val": "IC val",
                              "ic_test": "IC test", "t_pool": "t(train+val)", "included": "채택", "weight_norm": "가중치"})
        st.dataframe(V[["점수", "구성요소", "현재(0~1)", "IC train", "IC val", "IC test", "t(train+val)", "채택", "가중치"]]
                     .round(3), hide_index=True, width="stretch")
        st.caption("IC = 구성요소와 이후 20일 SPY 수익률의 비겹침 순위상관. 방향은 사전 고정(위험자산에 우호적일수록 큼)이며 "
                   "train·val 모두 그 방향이고 t≥2일 때만 채택. 지금은 채택 0건. 참고: VIX·낙폭 계열은 세 구간 모두 "
                   "반대(공포가 클수록 이후 수익률이 좋음) 방향이지만, 이미 본 데이터이므로 채택하지 않고 "
                   "앞으로의 실제 데이터로 재검증할 가설로만 둡니다.")

st.divider()

# ───────────────────────── 2. WALL STREET CAPITAL ROTATION ─────────────────────────
st.header("WALL STREET CAPITAL ROTATION")
sec, ind = rd("sector_snapshot.parquet"), rd("industry_snapshot.parquet")


def rot_table(df):
    d = df.copy()
    for c in ("rs_3d", "rs_1w", "rs_1m", "cmf_1w", "cmf_1m"):
        d[c] = d[c] * 100
    show = d[["name", "score", "score_price", "score_flow", "direction", "flow_dir", "rs_1m", "rs_1w", "rs_3d",
              "cmf_1m", "cmf_1w"]].rename(columns={
        "name": "섹터", "score": "합성", "score_price": "가격", "score_flow": "자금", "direction": "가격방향",
        "flow_dir": "자금방향", "rs_1m": "1M RS%", "rs_1w": "1W RS%", "rs_3d": "3D RS%",
        "cmf_1m": "1M 자금%p", "cmf_1w": "1W 자금%p"})
    pb = lambda t: st.column_config.ProgressColumn(t, min_value=0, max_value=100, format="%d")  # noqa: E731
    st.dataframe(show, hide_index=True, width="stretch", height=min(40 + 35 * len(show), 460),
                 column_config={"합성": pb("합성"), "가격": pb("가격"), "자금": pb("자금"),
                                **{c: st.column_config.NumberColumn(c, format="%+.2f") for c in
                                   ("1M RS%", "1W RS%", "3D RS%", "1M 자금%p", "1W 자금%p")}})


if sec is not None and len(sec):
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.subheader("GICS 11 섹터")
        rot_table(sec)
    with c2:
        st.subheader("회전 지도 (색=자금 점수, 크기=자금 강도)")
        fig = go.Figure()
        fig.add_hline(y=0, line_width=1, line_color="rgba(100,116,139,0.4)")
        fig.add_vline(x=0, line_width=1, line_color="rgba(100,116,139,0.4)")
        fig.add_scatter(x=sec.rs_1m * 100, y=sec.rs_1w * 100, mode="markers+text", text=sec.name,
                        textposition=["top center" if i % 2 == 0 else "bottom center" for i in range(len(sec))], marker=dict(size=11 + (sec.score_flow - 50).abs() / 5, color=sec.score_flow, colorscale="RdYlGn",
                                                               cmin=0, cmax=100), textfont=dict(size=10))
        for x, y, t in [(1, 1, "선도"), (-1, 1, "개선"), (-1, -1, "부진"), (1, -1, "약화")]:
            fig.add_annotation(xref="paper", yref="paper", x=0.5 + 0.47 * x, y=0.5 + 0.47 * y, text=t, showarrow=False,
                               font=dict(size=11, color="#94a3b8"))
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=8, b=0), xaxis_title="1개월 RS (%)",
                          yaxis_title="1주 RS (%)")
        st.plotly_chart(fig, width="stretch")
    with st.expander("산업/테마 (반도체·소프트웨어·방산·에너지 세부 등)"):
        if ind is not None and len(ind):
            rot_table(ind)
    st.markdown('<span class="badge no">스냅샷</span> <span class="small">가격 점수=SPY 대비 상대강도 순위, 자금 점수=SPY 대비 '
                '체이킨 자금흐름(CMF: 고저폭 안에서 종가 위치×거래량) 순위. ETF 설정·환매 자금유입은 무료 장기 데이터가 없어 '
                '가격·거래량 대리지표를 씁니다. 이 신호들의 이후 수익률 예측력은 백테스트 결과(아래 방법론)를 확인하세요.</span>',
                unsafe_allow_html=True)
    flow = feat("flow").dropna(how="all")
    cf = flow.iloc[-1]
    names_cf = {"Technology": "SP_Tech_E-mini", "Financials": "SP_Fin_E-mini", "Energy": "SP_Energy_E-mini",
                "Healthcare": "SP_Health_E-mini", "Industrials": "SP_Ind_E-mini", "Utilities": "SP_Util_E-mini",
                "ConsDisc": "SP_ConsDisc_E-mini", "ConsStaples": "SP_ConsStaples_E-mini",
                "Materials": "SP_Materials_E-mini"}
    rows = []
    for nm, key in names_cf.items():
        g = lambda f: cf.get(f"cftc__{key}__{f}", np.nan)  # noqa: E731
        rows.append({"섹터": nm, "자산운용 순포지션%OI": g("assetmgr_net") * 100, "백분위": g("assetmgr_net_rank") * 100,
                     "4주변화%p": g("assetmgr_net_chg4w") * 100, "투기(레버리지) 순포지션%OI": g("levmoney_net") * 100,
                     "투기 백분위": g("levmoney_net_rank") * 100, "투기 4주변화%p": g("levmoney_net_chg4w") * 100})
    with st.expander("월가 기관 포지셔닝 — S&P 섹터 E-mini 선물 (CFTC, 주간, 2017~)"):
        st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, width="stretch")
        st.caption("자산운용사(장기 기관)와 레버리지 펀드의 선물 순포지션(미결제약정 대비). 화요일 기준·금요일 공표라 최대 며칠 지연. "
                   "백분위는 해당 계약 이력 대비 위치. 개별 섹터 선물은 규모가 작아 해석에 주의.")

st.divider()

# ───────────────────────── 3. KOREA TRANSMISSION ─────────────────────────
st.header("KOREA TRANSMISSION")
kh = rd("korea_model_history.parquet")
km = S["korea_model"]
a, b = st.columns([1, 1.15])
with a:
    st.subheader("오늘 미국 마감 → 다음 한국 장")
    last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
    z = last.pred_foreign_z
    direction = "순매수 우위" if z > 0.15 else ("순매도 우위" if z < -0.15 else "중립")
    st.metric("예상 외국인 수급 (z)", f"{z:+.2f}", direction)
    st.progress(min(max(float(last.korea_score) / 100, 0.0), 1.0),
                text=f"역사 대비 위치 {last.korea_score:.0f}/100 (낮을수록 매도 쪽)")
    co = km["coef"]
    st.caption("회귀: 다음 한국 거래일 KOSPI 외국인 순매수 z = "
               + " + ".join(f"{v:+.2f}×{k}" for k, v in co.items() if k != "const") + " (입력은 표준화)")
    ev = km["eval"]
    st.markdown(f"**검증**: 적합 구간(2005~2022) 상관 train {ev['train']['corr']:.2f} · val {ev['val']['corr']:.2f}, "
                f"**표본외(2023~) {ev['test']['corr']:.2f}** (n={ev['test']['n']})  \n"
                f"설명력은 크지 않다(R² {ev['test']['r2_vs_mean']:.2f}) — 방향 힌트이지 정밀 예측이 아닙니다.")
with b:
    st.subheader("KOSPI 투자자별 순매수 (최근 120거래일 누적, 조원)")
    kp = PARQUET / "kis_investor_kospi.parquet"
    d = pd.read_parquet(kp).tail(120)
    fig = go.Figure()
    for c, n in [("frgn_ntby_tr_pbmn", "외국인"), ("orgn_ntby_tr_pbmn", "기관"), ("prsn_ntby_tr_pbmn", "개인")]:
        fig.add_scatter(x=d.index, y=(d[c] / 1e6).cumsum(), name=n, mode="lines")
    fig.update_layout(height=290, margin=dict(l=0, r=0, t=8, b=0), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("예측 vs 실제 (최근 거래일)")
    t = kh.dropna(subset=["pred_foreign_z"]).tail(11).copy()
    t["잔차"] = t.actual_next_foreign_z - t.pred_foreign_z
    t = t.rename(columns={"pred_foreign_z": "예측 z", "actual_next_foreign_z": "실제 z"})[["예측 z", "실제 z", "잔차"]]
    t.index = [f"{i.date()}(미국)" for i in t.index]
    st.dataframe(t.round(2), width="stretch")
    res = (kh.actual_next_foreign_z - kh.pred_foreign_z).dropna()
    r20 = res.tail(20).mean() / (res.loc[:"2022-12-31"].std())
    st.caption(f"최근 20일 평균 잔차 {r20:+.2f}σ → " + ("한국 외국인이 미국 신호보다 **약하게** 반응(괴리)" if r20 < -0.5 else
               "한국 외국인이 미국 신호보다 **강하게** 반응(괴리)" if r20 > 0.5 else "역사적 전달 패턴을 대체로 따르는 중")
               + " · 마지막 행은 아직 실제값이 없는 예측")
with c2:
    st.subheader("전달 시차 (상관 vs 지연일)")
    ll = rd("lead_lag.parquet")
    pair = st.selectbox("관계", ["SOXX → KOSPI_foreign_z", "QQQ → KOSPI_foreign_z", "SOXX → KR_Semis", "QQQ → KOSPI",
                                 "DXY → KOSPI_foreign_z"], label_visibility="collapsed")
    x, y = [s.strip() for s in pair.split("→")]
    q = ll[(ll.x == x) & (ll.y == y) & ll.split.isin(["train", "val", "test"])]
    fig = go.Figure()
    for sp, col in [("train", "#64748b"), ("val", "#3b82f6"), ("test", "#f59e0b")]:
        g = q[q.split == sp]
        fig.add_bar(x=[f"{l}D" for l in g.lag], y=g.r, name=sp)
    fig.update_layout(barmode="group", height=280, margin=dict(l=0, r=0, t=8, b=0), yaxis_title="상관계수",
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("미국 t일 마감 → 한국 t+L일. 1거래일에 집중되고 3일 이후엔 사라집니다. 상관은 인과가 아닙니다.")

st.subheader("한국 업종별 외국인·기관 자금흐름 (KIS 종목별 수급 합산)")
kr = feat("korea")
rowk = kr.dropna(how="all").iloc[-1]
names = {"Semis": "반도체", "Auto": "자동차", "Banks": "은행", "Healthcare": "헬스케어/바이오", "EnergyChem": "에너지화학",
         "Battery": "2차전지", "Shipbuilding": "조선", "Defense": "방산"}
tab = pd.DataFrame([{"업종": v,
                     "외국인 1D%": rowk.get(f"kr__{k}_foreign__ratio_1d", np.nan) * 100,
                     "외국인 5D%": rowk.get(f"kr__{k}_foreign__ratio_5d", np.nan) * 100,
                     "외국인 20D%": rowk.get(f"kr__{k}_foreign__ratio_20d", np.nan) * 100,
                     "기관 20D%": rowk.get(f"kr__{k}_inst__ratio_20d", np.nan) * 100,
                     "업종ETF 20D RS%": rowk.get(f"kr__KR_{k}__rs_20d", np.nan) * 100} for k, v in names.items()])
fig = go.Figure()
fig.add_bar(x=tab["업종"], y=tab["외국인 20D%"], name="외국인 20일")
fig.add_bar(x=tab["업종"], y=tab["기관 20D%"], name="기관 20일")
fig.update_layout(barmode="group", height=280, margin=dict(l=0, r=0, t=8, b=0), yaxis_title="순매수 / 거래대금 (%)",
                  legend=dict(orientation="h", y=1.15))
cA2, cB2 = st.columns([1, 1.1])
with cA2:
    st.plotly_chart(fig, width="stretch")
with cB2:
    st.dataframe(tab.round(2), hide_index=True, width="stretch")
st.caption("업종 대표 종목(3~4개)의 순매수대금 합 ÷ 거래대금 합. +면 해당 투자자가 순매수. 대표 종목은 현재 대형주로 고정(과거 구성과 다를 수 있음).")

st.divider()

# ───────────────────────── 4. HISTORICAL ANALOG ─────────────────────────
st.header("HISTORICAL ANALOG")
A = S["analog"]
st.markdown(f"### Similar historical regimes: **{A['n_episodes']}** 사례 ({A['n_days']}일)")
st.caption(f"현재 상태 벡터와의 거리 하위 10%(≤{A['threshold']:.2f}) · 가장 가까운 날 {A['nearest']:.2f} · "
           f"전체 중앙값 {A['median_all']:.2f} (RMS 거리, 0~1 척도) · 최근 130영업일은 후보에서 제외")
fw = rd("analog_forward.parquet")
assets = sorted(fw.asset.unique())
default = assets.index("SPY") if "SPY" in assets else 0
cA, cB = st.columns([1, 3])
with cA:
    asset = st.selectbox("자산", assets, index=default)
    kind = st.radio("기준", ["절대수익", "벤치마크 대비"], horizontal=False)
sub = fw[(fw.asset == asset) & (fw.kind == ("abs" if kind == "절대수익" else "rel"))].sort_values("h")
with cB:
    if len(sub):
        fig = go.Figure()
        xs = [f"{h}D" for h in sub.h]
        fig.add_bar(x=xs, y=(sub.p90 - sub.p10) * 100, base=sub.p10 * 100, name="유사국면 10~90%", opacity=.35)
        fig.add_scatter(x=xs, y=sub["median"] * 100, mode="markers", name="유사국면 중앙값",
                        marker=dict(size=11, color="#f59e0b"))
        fig.add_scatter(x=xs, y=sub.base_median * 100, mode="markers", name="무조건 중앙값",
                        marker=dict(size=9, symbol="diamond", color="#64748b"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=8, b=0), yaxis_title="이후 수익률 (%)",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
        t = sub.assign(**{"기간": [f"{h}D" for h in sub.h], "독립 사례": sub.n_eff, "관측일": sub.n,
                          "중앙값%": sub["median"] * 100, "평균%": sub["mean"] * 100, "10분위%": sub.p10 * 100,
                          "90분위%": sub.p90 * 100, "승률%": sub.win * 100, "기준 승률%": sub.base_win * 100})
        st.dataframe(t[["기간", "독립 사례", "관측일", "중앙값%", "평균%", "10분위%", "90분위%", "승률%", "기준 승률%"]].round(1),
                     hide_index=True, width="stretch")
st.caption("유사 국면의 이후 수익률 분포입니다. 독립 사례가 적으면(수십 일이 한 사례) 분포를 신뢰하지 마세요. "
           "한국 자산은 미국 신호 다음 거래일 종가 진입 기준입니다.")
with st.expander("현재 상태 벡터 · 가장 가까운 과거 날짜"):
    st.json(A["vector"])
    nb = rd("analog_neighbors.parquet")
    st.dataframe(nb[nb.is_nearest].sort_values("dist").head(10).assign(date=lambda x: x.date.dt.date)
                 [["date", "dist", "episode"]].round(3), hide_index=True)

st.divider()
with st.expander("방법론·한계·데이터 현황"):
    st.markdown("""
- **레짐→섹터 매핑은 지지되지 않음**: 검정 2,764건 중 3개 구간(2005–18/2019–22/2023~) 모두 재현 0.47%(우연 기대 ≈1.25%).
- **섹터 상대강도·모멘텀의 예측력 불안정**: 단면 IC가 분할마다 부호가 바뀜.
- **미국→한국 전달은 안정적**: 시차 ≈1거래일. 단 상관이며 인과 아님, 가격(상대수익)으로는 일관되지 않음.
- **미포함(무료 장기 데이터 없음)**: ETF Fund Flow, 옵션 포지셔닝, 13F, Economic Surprise, ISM.
- Look-ahead 방지: 발표시점(vintage) 기준, expanding 순위, `tests/test_no_lookahead.py`.
- 자세한 내용: docs/DESIGN.md · FEATURES.md · BACKTEST.md (GitHub gtjdbfla/macro-flow-dashboard)
""")
    cov_path = DATA_DIR / "coverage.json"
    if cov_path.exists():
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        rows = []
        for src, v in cov.items():
            for k, x in v.get("series", {}).items():
                rows.append({"source": src, "series": k, "start": x.get("start"), "end": x.get("end"), "n": x.get("n"),
                             "error": x.get("error")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)
