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

st.title("② AI 수익성")
_cl = agg.iloc[-1]
ui.purpose("AI 기업의 수익성은 어떻게 변해 왔고, 지금 어떤 상태이며, 앞으로 무엇을 보면 될까?",
           f"파는 쪽(클라우드 매출 <b>{pct(_cl.yoy*100)}</b>, 마진 {_cl.margin*100:.0f}%)은 좋아지고, 투자하는 쪽(Big4 CAPEX <b>{pct(B.capex_yoy*100)}</b> vs 매출 "
           f"<b>{pct(B.rev_yoy*100)}</b>)은 회수·조달 부담이 커지고 있습니다. 공시 재무는 회사 전체 기준이라 AI만의 수치는 아니며, 주가를 예측하지 않습니다.",
           [("과거", "투자·매출·이익·클라우드가 최근 3년 어떻게 변했나"),
            ("현재", "투자 → 클라우드 수요 → 이익·현금 → 조달 네 단계 상태 · 회사별 표"),
            ("다음", "예측이 아닌 두 도구: 가정 시나리오 · 대만 공급망 선행 지표")],
           badge("현황판 · 예측 아님", "warn"))
ui.freshness(meta)

_fp = load("views/ai_forecast.parquet")
if _fp is not None:
    _Pf = _fp.set_index("metric")
    _fc = _Pf.loc["fcf"]
    _fx = (f"지금 증가율이 이어지면 다음 12개월에 CAPEX <b>{_Pf.loc['capex'].proj/1e8:,.0f}억 달러</b>({_Pf.loc['capex'].growth*100:+.0f}%), 매출 {_Pf.loc['revenue'].proj/1e8:,.0f}억({_Pf.loc['revenue'].growth*100:+.0f}%), "
           f"영업현금흐름 {_Pf.loc['ocf'].proj/1e8:,.0f}억({_Pf.loc['ocf'].growth*100:+.0f}%), 잉여현금흐름은 {_fc.now_ttm/1e8:,.0f}억 → 약 {_fc.proj/1e8:,.0f}억 달러입니다.")
else:
    _fx = ""
ui.howto(
    info_md=(
        "**과거 — 투자와 수익이 어떻게 변해 왔나**\n"
        "- Microsoft·Alphabet·Amazon·Meta(Big4)와 Oracle·CoreWeave의 **분기 CAPEX·영업현금흐름·매출·영업이익**(SEC 공시 원문, **공시된 날 이후에만** 사용, 정정 전 최초 값), 2008년~.\n"
        "- 클라우드 3사(AWS·Google Cloud·Microsoft)의 **부문 매출·영업이익**(실적 발표문 표를 읽음).\n\n"
        "**현재 — 지금 상태는 어떤가 (4단계 체인)**\n"
        "- ① 투자: 12개월 CAPEX와 증가율 · ② 수요: 클라우드 매출·마진, **수주잔고(RPO)**, NVIDIA 매출, 대만 AI 공급망 월매출 · "
        "③ 이익·현금: 영업이익·FCF, CAPEX 1달러당 이익 증가 · ④ 조달: CAPEX/영업현금흐름, 빅테크 신규 채권 **가산금리**, CoreWeave 쿠폰.\n"
        "- 위 지표를 **사전에 고정한 규칙의 신호 10개**(경고·주의·긍정·강함)로 판정합니다. 종합 점수는 가중치를 검증할 표본이 없어 만들지 않았습니다.\n\n"
        "**알 수 없는 것**: 재무는 **회사 전체 기준**이라 AI만의 수치가 아니고(신뢰 등급 C), CAPEX의 **구성**(서버·건물·칩)은 공시에 없습니다. "
        "CDS·유통시장 채권 수익률, HBM 가격도 무료 데이터가 없습니다.\n\n"
        "<span class='muted'>갱신: 분기 실적 시즌(1·4·7·10월 말)에 새 공시를 확인하면 자동 재계산, 대만 월매출은 매월 10일경, 채권은 발행 시.</span>"
    ),
    pred_md=(
        "**1. 기준선 전망 (" + badge("단서", "warn") + ")**: \"최근 분기의 전년동기비 증가율이 최근 4분기 합계에 그대로 이어진다\"는 한 줄 규칙으로 **다음 12개월 합계**를 계산합니다. "
        "합계 그대로·최근 분기×4·연간 증가율 지속 등 5개 단순 방법을 Big4 분기 자료에서 같은 방식으로 비교해 가장 덜 치우친 것을 골랐고, 그 방법의 과거 오차를 **범위**로 씁니다 "
        "(매출 ±6%, 영업현금흐름 ±15%, 영업이익 ±22%, CAPEX ±23%). " + _fx + "\n\n"
        "**한계**: 4분기 창이 겹쳐 독립 표본이 약 9개이고, 방법을 5개 중 최선으로 골랐습니다. CAPEX는 2023년 이후 실제가 전망보다 약 15% 컸고(위쪽 위험), "
        "FCF 전망은 최근 실제보다 높게 나오는 경향이 있습니다. 추세가 꺾이는 시점(투자 철회, 금리 충격)은 못 잡습니다.\n\n"
        "**2. 시나리오 슬라이더**: 사용자가 CAPEX·영업현금흐름 증가율을 직접 넣으면 12개월 뒤 FCF와 CAPEX/영업현금흐름을 단순 곱셈으로 보여줍니다(감가상각·투자 시점 지연 미반영). 예측이 아니라 \"이 가정이면 이렇다\"입니다.\n\n"
        "**3. 선행 지표 체크리스트**: 대만 공급망 월매출, 분기 공시(10월 말), 신규 채권 가산금리를 다음에 확인할 항목으로 둡니다. 예측력이 확인된 지표는 아닙니다.\n\n"
        "**검증해 보고 탈락한 방법(" + badge("참고", "mid") + ")**: 수주잔고 격차(순위 상관 +0.03), NVIDIA 매출이 CAPEX를 앞선다(표본외 개선 불일치), "
        "증가율 평균회귀(1분기 앞에서 \"지속\"보다 나쁨), CAPEX 기반 감가상각 계산. **주가 예측**도 이벤트 스터디에서 CAPEX 가속·감속이 반도체 수익을 구분하지 못해 하지 않습니다."
    ),
)


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

# ── 과거 ────────────────────────────────────────────────────────────────────
ui.stage("과거", "투자와 수익은 어떻게 변해 왔나", "최근 12개월 기준 전년 대비")
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

# ── 현재 ────────────────────────────────────────────────────────────────────
ui.stage("현재", "지금 네 단계는 어떤 상태인가", "투자 → 수요 → 이익·현금 → 조달")
st.markdown(html, unsafe_allow_html=True)
st.markdown('<p class="note">상태 판정은 사전에 고정한 규칙의 신호 10개를 단계별로 묶은 것입니다(근거·방법 페이지). 종합 점수는 가중치를 검증할 표본이 없어 만들지 않았습니다.</p>',
            unsafe_allow_html=True)
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

# ── 다음 ────────────────────────────────────────────────────────────────────
ui.stage("다음", "앞으로 — 다음 12개월 기준선 전망과 두 가지 도구")
fp = load("views/ai_forecast.parquet")
fe = load("views/ai_forecast_eval.parquet")
fn = load("views/ai_forecast_negatives.parquet")
if fp is not None:
    P = fp.set_index("metric")
    _e = pd.Timestamp(P.loc["capex"].last_q_end)
    st.markdown(badge("기준선 전망", "info") + badge("단서", "warn") + f"**Big4 다음 12개월 ({(_e + pd.offsets.MonthBegin(1)):%Y-%m} ~ {(_e + pd.DateOffset(months=12)):%Y-%m})**", unsafe_allow_html=True)
    fc = P.loc["fcf"]
    st.markdown(f'<p class="lede">지금 증가율이 그대로 이어지면 다음 12개월에 Big4는 <b>CAPEX {P.loc["capex"].proj/1e8:,.0f}억 달러(+{P.loc["capex"].growth*100:.0f}%)</b>, '
                f'매출 {P.loc["revenue"].proj/1e8:,.0f}억(+{P.loc["revenue"].growth*100:.0f}%), 영업현금흐름 {P.loc["ocf"].proj/1e8:,.0f}억(+{P.loc["ocf"].growth*100:.0f}%)입니다. '
                f'그러면 <b>잉여현금흐름은 {fc.now_ttm/1e8:,.0f}억 → 약 {fc.proj/1e8:,.0f}억 달러</b>(범위 {fc.lo/1e8:,.0f} ~ {fc.hi/1e8:,.0f})로 0 부근까지 줄어듭니다. '
                f'가장 최근 분기({_e:%Y-%m})만 보면 이미 영업현금흐름 {P.loc["ocf"].last_q/1e8:,.0f}억 − CAPEX {P.loc["capex"].last_q/1e8:,.0f}억 = 약 {fc.last_q/1e8:,.0f}억 달러입니다.</p>', unsafe_allow_html=True)
    f1, f2 = st.columns([1.25, 1], gap="large")
    with f1:
        st.markdown(ui.forecast_table(fp), unsafe_allow_html=True)
        st.markdown('<p class="note"><b>방법</b>: 최근 분기의 전년동기비 증가율이 최근 4분기 합계에 그대로 이어진다고 가정. 5개 단순 방법을 같은 원점에서 비교해 가장 덜 치우친 것을 골랐고, '
                    '그 방법의 과거 오차(RMSE)를 범위 폭으로 씁니다. 4분기 창이 겹쳐 독립 표본은 약 9개라 범위는 거친 추정입니다. '
                    '<b>CAPEX는 2023년 이후 실제가 전망보다 약 15% 컸고</b>(위쪽 위험), FCF 전망은 최근 실제보다 높게 나오는 경향(12%p)이 있어 아래쪽 위험이 있습니다.</p>', unsafe_allow_html=True)
    with f2:
        names = ["CAPEX", "영업현금흐름", "잉여현금흐름"]
        keys = ["capex", "ocf", "fcf"]
        fig = go.Figure()
        fig.add_bar(name="최근 12개월", x=names, y=[P.loc[k].now_ttm / 1e8 for k in keys], marker_color="#8a99a3")
        fig.add_bar(name="다음 12개월 전망", x=names, y=[P.loc[k].proj / 1e8 for k in keys], marker_color="#5b8def",
                    error_y=dict(type="data", symmetric=False, array=[(P.loc[k].hi - P.loc[k].proj) / 1e8 for k in keys], arrayminus=[(P.loc[k].proj - P.loc[k].lo) / 1e8 for k in keys]))
        fig.update_layout(barmode="group")
        plot(fig, 300)
    if fn is not None:
        with st.expander("더 정교한 예측 변수는 기준선을 넘지 못했습니다 (수주잔고·NVIDIA 매출·평균회귀 모델)"):
            st.dataframe(pd.DataFrame({"검정": fn.test, "표본": fn.n, "결과": fn.result, "판정": fn.verdict, "이유": fn.why}), hide_index=True, width="stretch")
    st.markdown('<p class="note">아래 두 도구는 이 기준선 대신 <b>직접 가정</b>을 넣어 보는 시나리오(왼쪽)와, 다음에 확인될 선행 지표(오른쪽)입니다.</p>', unsafe_allow_html=True)

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
