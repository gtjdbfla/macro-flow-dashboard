"""홈: 맨 위에 두 질문의 답(요약 카드 한 쌍) → 아래에서 한쪽만 골라 자세히 본다.
스크롤을 줄이고(한 번에 한 주제), 글 대신 큰 숫자·표·칩으로 보여준다."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from ui import badge, callout, chips, duo, ev_card, kpis, level_kind, load, pct, plot, split_rows, summary_card, trio

meta, state = load("views/meta.json"), load("views/state.json")
sec = load("results/sector_snapshot.parquet")
kh = load("views/korea_model_recent.parquet")
quint = load("views/korea_quintile.parquet")
latest = load("results/ai_latest.parquet")
agg = load("results/ai_cloud_agg.parquet")
usd = load("views/sector_flow_usd.parquet")
nxt = load("views/flow_next.json")
ha = load("views/home_ai.parquet")
hd = load("views/sector_holdings.parquet")
cq = load("views/ai_capex_q.parquet")
ev = load("views/evidence.json")
fh = load("views/flow_history.parquet")
hist = load("results/ai_history.parquet")
tw = ui.load_parquet_dir("twse_monthly.parquet")
flags = json.loads(open(ui.DATA_DIR / "results/ai_flags.json", encoding="utf-8").read())["flags"] if (ui.DATA_DIR / "results/ai_flags.json").exists() else []

# ── 머리말 ───────────────────────────────────────────────────────────────────
st.title("Macro & Capital Flow")
st.markdown('<p class="lede">두 가지 질문에 <b>과거 → 현재 → 다음</b> 순서로 답합니다. '
            '① 월가의 돈은 어디에서 어디로 움직였고 지금 어디에 있나 · ② AI 기업의 수익성은 어떻게 변해 왔고 지금 어떤 상태인가. '
            '앞날을 맞히는 도구가 아니라, 확인된 것과 아직 확인되지 않은 것을 구분해 보여주는 현황판입니다.</p>', unsafe_allow_html=True)
ui.legend()
ui.freshness(meta)
stt = load("status.json")
if stt is not None:
    import datetime as _dt
    _age = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(stt["finished"])).total_seconds() / 3600
    if not stt["ok"]:
        st.warning("자동 갱신에 문제가 있습니다: " + "; ".join(stt["warnings"])[:300] + " (자세한 내용: 갱신 현황 페이지)")
    elif _age > 72:
        st.warning(f"자동 갱신이 {_age/24:.0f}일 전 이후 실행되지 않았습니다. 서버 크론을 확인하세요.")
st.caption(f"시장 상태 · {state['regime']} 국면(성장 {'가속' if state['growth']=='Accelerating' else '둔화'}, 물가 "
           f"{'가속' if state['inflation']=='Accelerating' else '둔화'}) · 정책 {state['policy']} · 유동성 {state['liquidity']} · 달러 {state['dollar']}"
           f" · {state['days_in_regime']}거래일째 유지")

# ── 공통 값 ──────────────────────────────────────────────────────────────────
etf = sec[sec.flow_src == "ETF유입"].copy().merge(usd, left_on="name", right_on="sector", how="left")
top = etf.sort_values("flow_1m", ascending=False).iloc[0]
bot_pct = etf.sort_values("flow_1m", ascending=True).iloc[0]
bot_usd = etf.sort_values("flow_20d_usd", ascending=True).iloc[0]
last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
z = float(last.pred_foreign_z)
zdir = "순매수 쪽" if z > 0.15 else ("순매도 쪽" if z < -0.15 else "중립")

B = latest[latest.ticker == "BIG4"].iloc[0]
cl = agg.iloc[-1]
fl = {f["id"]: f for f in flags}
demand = [i for i in ("F4", "F6", "F7", "F8") if i in fl]
press = [i for i in ("F1", "F2", "F3", "F9", "F10", "F11", "F12", "F13", "F14") if i in fl]
n_good = sum(level_kind(fl[i]["level"]) == "good" for i in demand)
n_bad = sum(level_kind(fl[i]["level"]) in ("bad", "warn") for i in press)
if n_good >= len(demand) / 2 and n_bad >= len(press) / 2:
    verdict = "AI 서비스를 파는 쪽(클라우드·공급망)은 좋아지고, 투자하는 쪽의 현금흐름과 조달 부담은 커지고 있습니다."
elif n_good >= len(demand) / 2:
    verdict = "수요와 클라우드 수익이 좋아지고 있고, 투자 부담 신호는 아직 제한적입니다."
else:
    verdict = "수요 쪽 신호가 약해지고 있어 투자 회수 속도를 주의해서 볼 필요가 있습니다."

# ── 요약 카드 한 쌍 ──────────────────────────────────────────────────────────
head1 = (f"최근 20일 자금은 <b>{top['name']}</b>으로 가장 많이 들어오고, 금액으로는 <b>{bot_usd['name']}</b>, 비율로는 <b>{bot_pct['name']}</b>에서 가장 많이 빠졌습니다. "
         f"다음 한국 장 외국인은 <b>{zdir}</b>으로 보입니다.")
slt_ = load("views/ai_safety_latest.parquet")
extra = ""
if slt_ is not None:
    _b = slt_.set_index("ticker")
    extra = (f" 빚 쪽은 <b>Oracle 런웨이 {_b.loc['ORCL','runway_y']:.1f}년, CoreWeave {_b.loc['CRWV','runway_y']:.1f}년</b>"
             f"(현금 ÷ FCF 적자·12개월 만기)이라 신규 조달이 필요한 구조입니다.")
head2 = f"{verdict}{extra} 수요 신호 {n_good}/{len(demand)}개 긍정, 회수·조달 신호 {n_bad}/{len(press)}개 경고·주의."
c1 = summary_card("① 월가의 자금은 어디에 있나", badge("한국 전달만 검증됨", "info") + badge("섹터 이동은 약한 단서", "warn"), head1,
                  [("순유입 1위 (20일)", top["name"], f"순자산의 {top.flow_1m*100:+.1f}% · {top.flow_20d_usd/1e8:+,.0f}억 달러"),
                   ("가장 많이 빠진 곳", bot_usd["name"], f"{bot_usd.flow_20d_usd/1e8:+,.0f}억 달러 · 순자산의 {bot_usd.flow_1m*100:+.1f}%"),
                   ("한국 다음 장 외국인", f"{z:+.2f}", zdir)])
c2 = summary_card("② AI 수익성은 어떻게 변하고 있나", badge("현황판 · 예측 아님", "warn"), head2,
                  [("투자 vs 매출 증가", f"{pct(B.capex_yoy*100)} / {pct(B.rev_yoy*100)}", "Big4 CAPEX / 매출, 12개월"),
                   ("클라우드 3사", f"{pct(cl.yoy*100)}", f"매출 증가 · 영업이익률 {cl.margin*100:.0f}% (1년 전 {cl.margin_prior*100:.0f}%)"),
                   ("잉여현금흐름", f"{B.fcf_ttm/1e8:,.0f}억 달러", f"전년 대비 {B.fcf_chg/1e8:+,.0f}억 달러")])
duo(c1, c2)


# ── 상세 (선택한 쪽만 그린다) ─────────────────────────────────────────────────
@st.fragment
def detail():
    pick = st.segmented_control("자세히 보기", ["① 월가 자금", "② AI 수익성"], default="① 월가 자금", key="home_detail", label_visibility="collapsed")
    if pick == "② AI 수익성":
        ai_detail()
    else:
        flow_detail()


def flow_detail():
    ui.stage("과거", "최근 1년, 섹터별 자금은 어떻게 움직였나", "20일 순유입(%순자산) · 위 띠는 매크로 국면")
    d = fh.tail(52)
    secs = [c for c in d.columns if c != "regime"]
    codes = {"Goldilocks": 0, "Reflation": 1, "Slowdown": 2, "Stagflation": 3}
    colors = ["#22a55a", "#e0b41e", "#5b8def", "#ee5555"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.07, 0.93], vertical_spacing=0.015)
    fig.add_trace(go.Heatmap(z=[d["regime"].map(codes).astype(float).values], x=d.index, colorscale=[[i / 3, c] for i, c in enumerate(colors)], zmin=0, zmax=3,
                             showscale=False, text=[d["regime"].fillna("").values], hovertemplate="%{x|%Y-%m-%d} %{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=(d[secs].T * 100).clip(-10, 10).values, x=d.index, y=secs, colorscale="RdYlGn", zmid=0, zmin=-10, zmax=10,
                             colorbar=dict(title="%", len=0.7, thickness=10), hovertemplate="%{y} %{x|%Y-%m-%d}: %{z:+.1f}%<extra></extra>"), row=2, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    plot(fig, 360, legend=False)
    st.markdown('<p class="note">초록=자금 유입, 빨강=유출. 위 띠 색: 초록 Goldilocks · 노랑 Reflation · 파랑 Slowdown · 빨강 Stagflation. '
                '더 긴 기간(2007~)은 <b>① 월가 자금</b> 페이지에서 봅니다.</p>', unsafe_allow_html=True)

    ui.stage("현재", "지금 자금은 어디로 들어오고 어디서 빠졌나", "최근 20일, 실제 ETF 설정·환매")
    a, b = st.columns([1.35, 1], gap="large")
    with a:
        d = etf.sort_values("flow_20d_usd")
        fig = go.Figure(go.Bar(x=d.flow_20d_usd / 1e8, y=d["name"], orientation="h", marker_color=np.where(d.flow_20d_usd >= 0, "#22a55a", "#ee5555"),
                               text=[f"{v/1e8:+,.0f}억  ({p*100:+.1f}%)" for v, p in zip(d.flow_20d_usd, d.flow_1m)], textposition="outside",
                               hovertemplate="%{y}: %{x:+,.0f}억 달러<extra></extra>"))
        fig.update_xaxes(range=[d.flow_20d_usd.min() / 1e8 * 2.0 - 4, d.flow_20d_usd.max() / 1e8 * 1.7 + 4], title="억 달러 (괄호는 해당 ETF 순자산 대비 %)")
        plot(fig, 400, legend=False)
    with b:
        ins = etf.sort_values("flow_20d_usd", ascending=False).head(3)
        outs = etf.sort_values("flow_20d_usd").head(3)
        rows = lambda df: "".join(f"<tr><td>{r['name']}</td><td class='r'>{r.flow_20d_usd/1e8:+,.0f}억</td><td class='r'>{r.flow_1m*100:+.1f}%</td></tr>" for _, r in df.iterrows())  # noqa: E731
        st.markdown("**가장 많이 들어온 3곳**", unsafe_allow_html=True)
        st.markdown(f"<table class='mini'><tr><th>섹터</th><th class='r'>20일 순자금</th><th class='r'>순자산 대비</th></tr>{rows(ins)}</table>", unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>**가장 많이 빠진 3곳**", unsafe_allow_html=True)
        st.markdown(f"<table class='mini'><tr><th>섹터</th><th class='r'>20일 순자금</th><th class='r'>순자산 대비</th></tr>{rows(outs)}</table>", unsafe_allow_html=True)
        st.markdown('<p class="note" style="margin-top:10px">금액은 큰 섹터(Technology·Financials)가, 비율은 작은 섹터(Materials·Utilities)가 두드러집니다. '
                    '두 기준 모두 보세요.</p>', unsafe_allow_html=True)

    if hd is not None:
        st.markdown("#### 섹터별 대표 종목")
        st.markdown(ui.holdings_table(hd, list(etf.sort_values("flow_20d_usd", ascending=False)["name"])), unsafe_allow_html=True)
        st.markdown(f'<p class="note">섹터 ETF(SPDR) 공개 보유 종목 기준 {hd.as_of.iloc[0]:%Y-%m-%d}. 순서는 위 표와 같은 20일 순유입 금액순입니다. '
                    '시가총액 비중이라 소수 대형주가 섹터 움직임을 좌우합니다.</p>', unsafe_allow_html=True)
    ui.stage("다음", "다음에 자금이 몰릴 곳 후보", "과거 백테스트 기준 · 근거 등급을 함께 봅니다")
    T = load("views/evidence_tests.parquet")
    en = load("views/evidence_now.json")
    k2 = load("views/korea_model_v2.parquet")
    P = T[T.primary].set_index("claim")
    ru = P.loc["us_flow_contrarian"]
    card1 = ev_card(badge(ru.grade, ui.GRADE_KIND[ru.grade]), "자금이 몰린 섹터는 이후 20일 덜 오르는 경향", f"t {ru.all_t:+.1f}",
                    f"11개 섹터를 매일 비교한 상관(IC), 20일 후 SPY 대비 · 전체 {int(ru.all_n):,}일", split_rows(ru, "{:+.3f}", -1),
                    chips("지금 부진 가능(자금 쏠림)", [n for n, _ in nxt["in_top"]], "dn") + chips("지금 유리 가능(자금 이탈)", [n for n, _ in nxt["out_top"]], "up"))
    rf = P.loc["factor_momentum"]
    fac = {k: v for k, v in en["factor_12m"].items() if k != "Mom"}
    win = [f"{ui.FACTOR_KO[k]} {v*100:+.0f}%" for k, v in sorted(fac.items(), key=lambda kv: -kv[1]) if v > 0]
    los = [f"{ui.FACTOR_KO[k]} {v*100:+.0f}%" for k, v in sorted(fac.items(), key=lambda kv: kv[1]) if v <= 0]
    card2 = ev_card(badge(rf.grade, ui.GRADE_KIND[rf.grade]), "다음에 몰릴 팩터: 최근 12개월 강했던 팩터가 이어서 강한 경향", f"t {rf.all_t:+.1f}",
                    f"과거 12개월 부호 → 다음 달 수익, 5개 팩터 평균 · 월 평균 {rf.all_mean*100:+.2f}% · 1963~", split_rows(rf, "{:+.4f}", 1),
                    chips(f"지금 이어질 가능성(12개월 +, {en['asof'][:7]} 기준)", win or ["없음"], "up") + chips("지금 약세 이어질 가능성(12개월 −)", los or ["없음"], "dn"))
    q = quint[quint.period == "oos"]
    kt = k2[k2.set == "C +직전 외국인 z"].iloc[0]
    card3 = ev_card(badge("검증됨", "good"), "미국 마감 + 직전 외국인 수급 → 다음 한국 거래일 외국인 수급", f"{z:+.2f}",
                    f"모델값(z) · 지금은 {zdir} (표본외 상관 {ev['korea_eval']['test']['corr']:.2f}, 설명력 {ev['korea_eval']['test']['r2_vs_mean']*100:.0f}%)",
                    [("예측 최상위 구간", "순매수 확률", f"{q.buy_prob.iloc[-1]:.0f}%"), ("예측 최하위 구간", "순매수 확률", f"{q.buy_prob.iloc[0]:.0f}%"),
                     ("평소", "순매수 확률", "약 47%")],
                    '<div class="chipl">효과는 거의 시가 갭에서 나오고 개장 후 초과수익은 확인되지 않았습니다.</div>')
    trio(card1, card2, card3)
    st.markdown(ui.grade_table(T), unsafe_allow_html=True)
    st.markdown('<p class="note">구간: 미국 ETF·한국 종목은 2005–18 / 2019–22 / 2023~, 산업 모멘텀은 1927–62 / 1963–99 / 2000~, 팩터 모멘텀은 1963–89 / 1990–2009 / 2010~. '
                '등급 규칙: <b>검증됨</b> = 세 구간 같은 부호 + 2구간 이상 |t|≥2 + 전체 |t|≥3, <b>단서</b> = 전체 |t|≥2 + 2구간 이상 같은 부호, 나머지 <b>참고</b>. '
                '호라이즌이 여럿이면 사전에 정한 대표 하나만 판정에 씁니다.</p>', unsafe_allow_html=True)
    callout("<b>이 근거들은 이미 본 과거 데이터에 대한 평가입니다.</b> 전향 장부(근거·방법 페이지)에 사전 등록해 앞으로 쌓이는 실제 데이터로 다시 확인합니다. "
            "팩터·산업 모멘텀은 <b>가격 추세</b>이지 자금 유입 데이터가 아니며, 학계에 알려진 뒤 약해지는 경향이 있습니다(산업 모멘텀은 2000년 이후 약화). "
            "한국 외국인 순매수 지속은 종목 단위로 다시 봐도 확인되지 않아 <b>참고</b>로 내렸고, 국면·상대강도로 섹터를 맞히는 것도 우연 수준입니다.")


def ai_detail():
    ui.stage("과거", "투자와 수익은 어떻게 변해 왔나", "최근 12개월 기준 전년 대비, 분기마다 갱신")
    b4h = hist[hist.ticker == "BIG4"].dropna(subset=["capex_yoy"]).sort_values("period_end").tail(12)
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("**투자는 매출·이익보다 빠르게 늘었다** (Big4, 전년 대비 %)")
        fig = go.Figure()
        for c, n, col in [("capex_yoy", "CAPEX", "#f97316"), ("rev_yoy", "매출", "#5b8def"), ("opinc_yoy", "영업이익", "#22a55a")]:
            fig.add_scatter(x=b4h.period_end, y=b4h[c] * 100, name=n, mode="lines+markers", line=dict(color=col, width=2.4))
        plot(fig, 300)
    with p2:
        st.markdown("**클라우드 매출은 가속하고 마진도 오른다** (3사 합계)")
        ca = agg.tail(10)
        fig = go.Figure()
        fig.add_bar(x=ca.quarter_end, y=ca.yoy * 100, name="매출 증가율(%)", marker_color="#5b8def", opacity=.8)
        fig.add_scatter(x=ca.quarter_end, y=ca.margin * 100, name="영업이익률(%)", mode="lines+markers", line=dict(color="#f59e0b", width=2.4))
        plot(fig, 300)

    ui.stage("현재", "누가 얼마를 투자하고, 얼마나 벌어들이나", "회사별 12개월 CAPEX와 변화")
    t = ha.copy()
    tab = pd.DataFrame({"회사": t["name"], "CAPEX (억 달러)": t.capex / 1e8, "비중": t.share * 100, "1년간 증가액 (억 달러)": t.capex_delta / 1e8,
                        "증가 기여": t.delta_share * 100, "증가율": t.capex_yoy * 100, "증가율 변화 (%p, 2분기 전 대비)": (t.capex_yoy - t.capex_yoy_2q) * 100,
                        "CAPEX / 영업현금흐름": t.capex_ocf, "2분기 전": t.capex_ocf_2q})
    st.dataframe(tab, hide_index=True, width="stretch", height=250, column_config={
        "CAPEX (억 달러)": st.column_config.NumberColumn(format="%.0f"), "비중": st.column_config.NumberColumn(format="%.0f%%"),
        "1년간 증가액 (억 달러)": st.column_config.NumberColumn(format="%+.0f"), "증가 기여": st.column_config.NumberColumn(format="%.0f%%"),
        "증가율": st.column_config.NumberColumn(format="%.0f%%"), "증가율 변화 (%p, 2분기 전 대비)": st.column_config.NumberColumn(format="%+.0f"),
        "CAPEX / 영업현금흐름": st.column_config.NumberColumn(format="%.2f"), "2분기 전": st.column_config.NumberColumn(format="%.2f")})
    by_capex, by_delta = t.sort_values("capex", ascending=False), t.sort_values("capex_delta", ascending=False)
    worse = [r["name"] for _, r in t.iterrows() if r.capex_ocf > r.capex_ocf_2q + 0.03]
    better = [r["name"] for _, r in t.iterrows() if r.capex_ocf < r.capex_ocf_2q - 0.03]
    st.markdown(f"- **가장 많이 쓰는 곳**: {by_capex.iloc[0]['name']} ({by_capex.iloc[0].share*100:.0f}%) · **1년간 가장 많이 늘린 곳**: {by_delta.iloc[0]['name']}, {by_delta.iloc[1]['name']} · "
                f"**가장 빠르게 늘리는 곳**: {t.sort_values('capex_yoy', ascending=False).iloc[0]['name']} (+{t.capex_yoy.max()*100:.0f}%) · "
                f"Oracle의 12개월은 {t[t.ticker == 'ORCL'].period_end.iloc[0][:7]}까지(결산월 차이)\n"
                f"- **영업현금흐름 대비 부담이 2분기 전보다 커진 곳**: {', '.join(worse) or '없음'} · **줄어든 곳**: {', '.join(better) or '없음'}")

    st.markdown("#### 1년 전보다 얼마를 더 쓰고, 얼마를 더 벌었나")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**분기별 CAPEX (억 달러)**")
        cqq = cq.copy()
        # Oracle은 5월 결산이라 분기말이 8·11·2·5월 → 가장 가까운 달력 분기말(9·12·3·6월)로 맞춰 다른 회사와 같은 막대에 쌓는다
        cqq["qe"] = cqq.end + pd.offsets.QuarterEnd(0)
        # Big4가 모두 공시하지 않은 마지막 분기는 일부 회사만 있어 오해를 부르므로 그리지 않는다
        full = cqq[cqq.ticker.isin(["AMZN", "GOOGL", "MSFT", "META"])].groupby("qe").ticker.nunique()
        cqq = cqq[cqq.qe.isin(full[full == 4].index)]
        cqq["q"] = cqq.qe.dt.strftime("%y.%m")
        fig = go.Figure()
        for tk, nm in [("AMZN", "Amazon"), ("GOOGL", "Alphabet"), ("MSFT", "Microsoft"), ("META", "Meta"), ("ORCL", "Oracle"), ("CRWV", "CoreWeave")]:
            g = cqq[cqq.ticker == tk]
            if len(g):
                fig.add_bar(x=g.q, y=g.value / 1e8, name=nm)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(type="category")
        plot(fig, 330)
        st.markdown('<p class="note">Oracle은 결산월이 달라(분기말 8·11·2·5월) 가장 가까운 달력 분기에 맞췄습니다(약 1개월 차이). '
                    'Big4가 모두 공시한 분기까지만 표시합니다.</p>', unsafe_allow_html=True)
    with b:
        b4 = ha[ha.ticker.isin(["MSFT", "GOOGL", "AMZN", "META"])]
        items = [("CAPEX (쓴 돈)", b4.capex_delta.sum(), "#f97316"), ("영업현금흐름", b4.ocf_delta.sum(), "#5b8def"), ("매출", b4.rev_delta.sum(), "#8a99a3"),
                 ("영업이익", b4.op_delta.sum(), "#22a55a"), ("잉여현금흐름", b4.fcf_delta.sum(), "#ee5555")]
        st.markdown("**Big4, 12개월 증가분 (억 달러)**")
        fig = go.Figure(go.Bar(x=[v / 1e8 for _, v, _ in items], y=[n for n, _, _ in items], orientation="h", marker_color=[c for _, _, c in items],
                               text=[f"{v/1e8:+,.0f}" for _, v, _ in items], textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[min(0, min(v for _, v, _ in items) / 1e8 * 1.6), max(v for _, v, _ in items) / 1e8 * 1.25])
        plot(fig, 330, legend=False)
    d_cap, d_ocf, d_op, d_fcf = items[0][1], items[1][1], items[3][1], items[4][1]
    st.markdown(kpis([("새로 쓴 돈 (CAPEX 증가)", f"{d_cap/1e8:,.0f}억", "달러, 1년 전 대비"),
                      ("새로 번 영업이익 / 영업현금흐름", f"{d_op/d_cap*100:.0f}% / {d_ocf/d_cap*100:.0f}%", "CAPEX 증가 대비"),
                      ("잉여현금흐름 변화", f"{d_fcf/1e8:+,.0f}억", "달러")]), unsafe_allow_html=True)

    st.markdown("#### 투자를 받는 쪽")
    nv = latest[latest.ticker == "NVDA"].iloc[0]
    tiles = [("NVIDIA 매출", pct(nv.rev_yoy * 100), "최근 12개월, 전년 대비")]
    if tw is not None and len(tw):
        tsmc = tw[tw.ticker == "2330"].sort_values("month").iloc[-1]
        tiles.append(("TSMC 월매출", pct(tsmc.yoy), f"{tsmc.month.strftime('%Y-%m')}, 전년 동월 대비"))
    tiles.append(("클라우드 3사 매출", pct(cl.yoy * 100), "최근 분기, 전년 동기 대비"))
    st.markdown(kpis(tiles), unsafe_allow_html=True)
    ui.stage("다음", "다음 12개월 기준선 전망과 지켜볼 것", "지금 증가율이 이어진다는 단순 가정 · 가정 시나리오는 ② AI 수익성 페이지")
    fp = load("views/ai_forecast.parquet")
    if fp is not None:
        st.markdown(ui.forecast_table(fp), unsafe_allow_html=True)
        f_ = fp.set_index("metric").loc["fcf"]
        st.markdown(f'<p class="note">최근 분기 전년동기비 증가율이 최근 4분기 합계에 이어진다는 방법(과거 오차를 범위로 사용). <b>잉여현금흐름은 {f_.now_ttm/1e8:,.0f}억 → 약 {f_.proj/1e8:,.0f}억 달러</b>로 '
                    '0 부근까지 줄어드는 쪽이며, CAPEX는 최근 실제가 전망보다 컸습니다(위쪽 위험). 근거 등급은 단서입니다.</p>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**지금 경고·주의 신호**")
        watch = [f for f in flags if level_kind(f["level"]) in ("bad", "warn")]
        for f in watch:
            st.markdown(badge(f["level"], level_kind(f["level"])) + f"**{f['title']}**", unsafe_allow_html=True)
        if not watch:
            st.write("현재 경고·주의 신호 없음")
    with b:
        st.markdown("**다음에 확인될 것**")
        st.markdown("- 미국 분기 실적 공시(10월 말): CAPEX 증가율과 매출 증가율의 격차, 잉여현금흐름\n"
                    "- 대만 AI 공급망 월매출(매월 10일경): 증가율 20% 아래 회사가 늘어나는지\n"
                    "- 빅테크 신규 채권 발행 가산금리: 직전보다 더 넓어지는지")
    callout("투자 내역(서버·건물·칩 등 구성)은 공시에 세부가 없어 알 수 없고, 위 CAPEX는 <b>회사 전체</b> 수치입니다. 이 페이지는 현재 상태와 방향을 보여줄 뿐 주가를 예측하지 않습니다.", "info")


detail()
st.markdown('<p class="note">상단 메뉴: <b>① 월가 자금</b>(과거·현재·다음 전체), <b>② AI 수익성</b>(변화·시나리오), <b>갱신 현황</b>, <b>근거·방법</b>.</p>', unsafe_allow_html=True)
