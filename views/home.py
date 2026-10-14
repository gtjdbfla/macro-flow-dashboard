"""홈: 두 질문의 답을 한 화면에 — 요약 카드 아래에 '무엇이, 얼마나, 어떻게 변하고 있는지'를 바로 이어서 보여준다."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import badge, card, level_kind, load, pct, plot

meta, state = load("views/meta.json"), load("views/state.json")
S = load("results/summary.json")
sec = load("results/sector_snapshot.parquet")
cf = load("views/cftc_now.parquet")
kh = load("views/korea_model_recent.parquet")
latest = load("results/ai_latest.parquet")
agg = load("results/ai_cloud_agg.parquet")
usd = load("views/sector_flow_usd.parquet")
nxt = load("views/flow_next.json")
ha = load("views/home_ai.parquet")
cq = load("views/ai_capex_q.parquet")
tw = ui.load_parquet_dir("twse_monthly.parquet")
flags = json.loads(open(ui.DATA_DIR / "results/ai_flags.json", encoding="utf-8").read())["flags"] if (ui.DATA_DIR / "results/ai_flags.json").exists() else []

st.title("Macro & Capital Flow")
st.markdown('<p class="lede">월가의 자금이 어디에 있고 어디로 움직여 왔는지, 그리고 AI 기업의 수익성이 어떻게 변하고 있는지를 데이터로 봅니다. '
            '앞날을 맞히는 도구가 아니라, 확인된 것과 아직 확인되지 않은 것을 구분해서 보여주는 현황판입니다.</p>', unsafe_allow_html=True)
ui.freshness(meta)
stt = load("status.json")
if stt is not None:
    import datetime as _dt
    _age = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(stt["finished"])).total_seconds() / 3600
    if not stt["ok"]:
        st.warning("자동 갱신에 문제가 있습니다: " + "; ".join(stt["warnings"])[:300] + " (자세한 내용: 갱신 현황 페이지)")
    elif _age > 72:
        st.warning(f"자동 갱신이 {_age/24:.0f}일 전 이후 실행되지 않았습니다. 서버 크론을 확인하세요.")
st.caption(f"시장 상태: {state['regime']} 국면(성장 {'가속' if state['growth']=='Accelerating' else '둔화'}, 물가 "
           f"{'가속' if state['inflation']=='Accelerating' else '둔화'}) · 정책 {state['policy']} · 유동성 {state['liquidity']} · 달러 {state['dollar']}"
           f" — {state['days_in_regime']}거래일째 유지")

# ════════════════════════ ① 월가 자금 ════════════════════════
etf = sec[sec.flow_src == "ETF유입"].copy().merge(usd, left_on="name", right_on="sector", how="left")
top, bot = etf.sort_values("flow_1m", ascending=False).iloc[0], etf.sort_values("flow_1m", ascending=True).iloc[0]
bot_usd = etf.sort_values("flow_20d_usd", ascending=True).iloc[0]
last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
z = float(last.pred_foreign_z)
zdir = "순매수 쪽" if z > 0.15 else ("순매도 쪽" if z < -0.15 else "중립")
head1 = (f"최근 20일 ETF 자금은 <b>{top['name']}</b>(순자산의 {top.flow_1m*100:+.1f}%, {top.flow_20d_usd/1e8:+,.0f}억 달러)로 가장 많이 들어오고, "
         f"비율로는 <b>{bot['name']}</b>({bot.flow_1m*100:+.1f}%), 금액으로는 <b>{bot_usd['name']}</b>({bot_usd.flow_20d_usd/1e8:+,.0f}억 달러)에서 가장 많이 빠졌습니다. "
         f"미국 마감 기준 다음 한국 장의 외국인은 <b>{zdir}</b>일 가능성이 높습니다.")
st.markdown("## ① 월가의 자금은 어디에 있나")
card("현재 위치", badge("한국 전달만 검증됨", "info") + badge("섹터 이동 후보는 약한 단서", "warn"), head1,
     [("순유입 1위(20일)", top["name"], f"순자산의 {top.flow_1m*100:+.1f}%"),
      ("가장 많이 빠진 곳", f"{bot_usd['name']} / {bot['name']}", "금액 기준 / 비율 기준"),
      ("한국 다음 장 외국인(z)", f"{z:+.2f}", zdir)])

a, b = st.columns([1.05, 1], gap="large")
with a:
    st.markdown("**섹터별 20일 순자금 (억 달러, 실제 ETF 설정·환매)**")
    d = etf.sort_values("flow_20d_usd")
    fig = go.Figure(go.Bar(x=d.flow_20d_usd / 1e8, y=d["name"], orientation="h", marker_color=np.where(d.flow_20d_usd >= 0, "#22a55a", "#ee5555"),
                           text=[f"{v/1e8:+,.0f}  ({p*100:+.1f}%)" for v, p in zip(d.flow_20d_usd, d.flow_1m)], textposition="outside",
                           hovertemplate="%{y}: %{x:+,.0f}억 달러<extra></extra>"))
    fig.update_xaxes(range=[d.flow_20d_usd.min() / 1e8 * 1.9 - 3, d.flow_20d_usd.max() / 1e8 * 1.6 + 3])
    plot(fig, 360, legend=False)
    st.markdown('<p class="note">괄호는 해당 섹터 ETF 순자산 대비 비율입니다. 금액은 큰 섹터(Technology·Financials)가, 비율은 작은 섹터가 두드러집니다.</p>', unsafe_allow_html=True)
with b:
    st.markdown("**과거 백테스트로 본 '다음에 자금이 몰릴 곳' 후보**")
    rules = pd.DataFrame(nxt["rules"])
    u = rules[rules.rule == "us_flow"].set_index("split") if len(rules) else None
    if u is not None and "all" in u.index:
        ua = u.loc["all"]
        tr, vl, te = (u.loc[s] if s in u.index else None for s in ("train", "val", "test"))
        st.markdown(badge("단서 · 약함", "warn") + "**자금이 몰린 섹터는 이후 덜 오르는 경향**", unsafe_allow_html=True)
        st.markdown(f"규칙: 최근 20일 순유입 상위 3개 섹터와 하위 3개 섹터의 이후 20일 SPY 대비 수익 비교(비겹침 {int(ua.n)}번). 상위 3개가 하위 3개보다 "
                    f"평균 <b>{abs(ua.mean_spread):.2f}%p {'덜' if ua.mean_spread < 0 else '더'} 올랐고</b>, 하위가 더 오른 비율은 <b>{ua.hit:.0f}%</b>였습니다. "
                    f"구간별로는 2005-18 {tr.mean_spread:+.2f}%p(t {tr.t:+.1f}), 2019-22 {vl.mean_spread:+.2f}%p(t {vl.t:+.1f}), 2023~ {te.mean_spread:+.2f}%p(t {te.t:+.1f})로 "
                    "<b>한 구간(2019-22)에서는 반대</b>였습니다.", unsafe_allow_html=True)
        ins = ", ".join(f"{n}" for n, _ in nxt["in_top"])
        outs = ", ".join(f"{n}" for n, _ in nxt["out_top"])
        st.markdown(f"**지금 값에 적용하면** — 상대적으로 부진 가능: <b>{ins}</b> (자금 쏠림) · 상대적으로 유리 가능: <b>{outs}</b> (자금 이탈)", unsafe_allow_html=True)
    k = rules[rules.rule == "kr_foreign"].set_index("split") if len(rules) else None
    if k is not None and "all" in k.index:
        ka = k.loc["all"]
        st.markdown(badge("참고 · 유의하지 않음", "mid") + "**한국: 외국인이 산 업종은 이후에도 약간 우위**", unsafe_allow_html=True)
        st.markdown(f"외국인 60일 순매수 비중 상위 3개 업종이 하위 3개보다 이후 20일 KOSPI 대비 평균 {ka.mean_spread:+.2f}%p(t {ka.t:+.1f}, {int(ka.n)}번)였지만 통계적으로 구분되지 않습니다. "
                    f"현재 상위: <b>{', '.join(n for n, _ in nxt['kr_in_top'])}</b> · 하위: <b>{', '.join(n for n, _ in nxt['kr_out_top'])}</b>", unsafe_allow_html=True)
    st.markdown(badge("검증됨", "good") + f"**한국 다음 장: 외국인 {zdir}** (모델값 {z:+.2f}, 표본외 상관 0.33)", unsafe_allow_html=True)
    st.markdown('<p class="note">이 후보들은 <b>예측이 아니라 약한 과거 패턴을 지금 값에 적용한 것</b>입니다. 적중 비율이 57%라는 건 동전 던지기보다 조금 낫다는 정도입니다. '
                '매크로 국면이나 상대강도로 섹터를 맞히는 것은 백테스트에서 우연 수준이었습니다. 스타일 팩터(모멘텀·가치·퀄리티 등)는 무료 자금유입 데이터가 없어 아직 검증하지 못했습니다.</p>',
                unsafe_allow_html=True)

st.divider()

# ════════════════════════ ② AI 수익성 ════════════════════════
B = latest[latest.ticker == "BIG4"].iloc[0]
cl = agg.iloc[-1]
fl = {f["id"]: f for f in flags}
demand = [i for i in ("F4", "F6", "F7", "F8") if i in fl]
press = [i for i in ("F1", "F2", "F3", "F9", "F10") if i in fl]
n_good = sum(level_kind(fl[i]["level"]) == "good" for i in demand)
n_bad = sum(level_kind(fl[i]["level"]) in ("bad", "warn") for i in press)
if n_good >= len(demand) / 2 and n_bad >= len(press) / 2:
    verdict = "AI 서비스를 파는 쪽(클라우드·공급망)은 좋아지고 있고, 투자를 하는 쪽의 현금흐름과 조달 부담은 커지고 있습니다."
elif n_good >= len(demand) / 2:
    verdict = "수요와 클라우드 수익이 좋아지고 있고, 투자 부담 신호는 아직 제한적입니다."
else:
    verdict = "수요 쪽 신호가 약해지고 있어 투자 회수 속도를 주의해서 볼 필요가 있습니다."
head2 = f"{verdict} 수요 신호 {n_good}/{len(demand)}개가 긍정, 회수·조달 신호 {n_bad}/{len(press)}개가 경고·주의입니다."
st.markdown("## ② AI 수익성은 어떻게 변하고 있나")
card("현재 위치", badge("현황판(예측 아님)", "warn"), head2,
     [("Big4 투자 vs 매출 증가", f"{pct(B.capex_yoy*100)} / {pct(B.rev_yoy*100)}", "CAPEX / 매출, 최근 12개월"),
      ("클라우드 3사 매출·마진", f"{pct(cl.yoy*100)} · {cl.margin*100:.0f}%", f"1년 전 마진 {cl.margin_prior*100:.0f}%"),
      ("잉여현금흐름", f"{B.fcf_ttm/1e8:,.0f}억 달러", f"전년 대비 {B.fcf_chg/1e8:+,.0f}억 달러")])

st.markdown("### 누가 투자하나 — 회사별 CAPEX와 변화")
t = ha.copy()
tab = pd.DataFrame({"회사": t["name"], "12개월 CAPEX(억 달러)": t.capex / 1e8, "비중": t.share * 100, "1년 전 대비 증가액(억 달러)": t.capex_delta / 1e8,
                    "증가 기여": t.delta_share * 100, "증가율": t.capex_yoy * 100, "증가율 변화%p": (t.capex_yoy - t.capex_yoy_2q) * 100,
                    "CAPEX/영업현금흐름": t.capex_ocf, "2분기 전": t.capex_ocf_2q})
a, b = st.columns([1.6, 1], gap="large")
with a:
    st.dataframe(tab.round(1), hide_index=True, width="stretch", height=268, column_config={
        "비중": st.column_config.NumberColumn(format="%.0f%%"), "증가 기여": st.column_config.NumberColumn(format="%.0f%%"),
        "증가율": st.column_config.NumberColumn(format="%.0f%%"), "CAPEX/영업현금흐름": st.column_config.NumberColumn(format="%.2f"),
        "2분기 전": st.column_config.NumberColumn(format="%.2f")})
    by_capex = t.sort_values("capex", ascending=False)
    by_delta = t.sort_values("capex_delta", ascending=False)
    worse = [r["name"] for _, r in t.iterrows() if r.capex_ocf > r.capex_ocf_2q + 0.03]
    better = [r["name"] for _, r in t.iterrows() if r.capex_ocf < r.capex_ocf_2q - 0.03]
    st.markdown(f"**읽는 법**(증가율 변화%p는 2분기 전 대비) — 가장 많이 쓰는 곳은 <b>{by_capex.iloc[0]['name']}</b>({by_capex.iloc[0].share*100:.0f}%)이고, 1년간 늘린 금액은 "
                f"<b>{by_delta.iloc[0]['name']}</b>와 <b>{by_delta.iloc[1]['name']}</b>가 가장 큽니다. 가장 빠르게 늘리는 곳은 <b>{t.sort_values('capex_yoy', ascending=False).iloc[0]['name']}</b>"
                f"(+{t.capex_yoy.max()*100:.0f}%)입니다. CAPEX/영업현금흐름이 2분기 전보다 <b>나빠진 곳</b>: {', '.join(worse) or '없음'} · <b>좋아진 곳</b>: {', '.join(better) or '없음'}.",
                unsafe_allow_html=True)
with b:
    st.markdown("**분기별 CAPEX (억 달러)**")
    cqq = cq.copy()
    cqq["q"] = cqq.end.dt.strftime("%y.%m")
    fig = go.Figure()
    for tk, nm in [("AMZN", "Amazon"), ("GOOGL", "Alphabet"), ("MSFT", "Microsoft"), ("META", "Meta"), ("ORCL", "Oracle"), ("CRWV", "CoreWeave")]:
        g = cqq[cqq.ticker == tk]
        if len(g):
            fig.add_bar(x=g.q, y=g.value / 1e8, name=nm)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    plot(fig, 300)

st.markdown("### 변화량 — 새로 쓴 돈 vs 새로 번 돈 (Big4, 1년 전 대비 12개월 증가분, 억 달러)")
b4 = ha[ha.ticker.isin(["MSFT", "GOOGL", "AMZN", "META"])]
items = [("CAPEX(쓴 돈)", b4.capex_delta.sum(), "#f97316"), ("영업현금흐름", b4.ocf_delta.sum(), "#5b8def"), ("매출", b4.rev_delta.sum(), "#8a99a3"),
         ("영업이익", b4.op_delta.sum(), "#22a55a"), ("잉여현금흐름", b4.fcf_delta.sum(), "#ee5555")]
a, b = st.columns([1.1, 1], gap="large")
with a:
    fig = go.Figure(go.Bar(x=[v / 1e8 for _, v, _ in items], y=[n for n, _, _ in items], orientation="h", marker_color=[c for _, _, c in items],
                           text=[f"{v/1e8:+,.0f}" for _, v, _ in items], textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[min(0, min(v for _, v, _ in items) / 1e8 * 1.5), max(v for _, v, _ in items) / 1e8 * 1.25])
    plot(fig, 280, legend=False)
with b:
    d_cap, d_ocf, d_op, d_fcf = items[0][1], items[1][1], items[3][1], items[4][1]
    st.markdown(f"1년 전보다 **CAPEX는 {d_cap/1e8:,.0f}억 달러 더** 쓰는데, 영업현금흐름은 {d_ocf/1e8:+,.0f}억 달러, 영업이익은 {d_op/1e8:+,.0f}억 달러 늘었습니다. "
                f"새로 쓴 돈 대비 새로 번 영업이익은 <b>{d_op/d_cap*100:.0f}%</b>, 영업현금흐름 증가는 <b>{d_ocf/d_cap*100:.0f}%</b>입니다. "
                f"그 차이만큼 잉여현금흐름이 <b>{d_fcf/1e8:+,.0f}억 달러</b> 변했습니다.", unsafe_allow_html=True)
    nv = latest[latest.ticker == "NVDA"].iloc[0]
    lines = [f"- **받는 쪽**: NVIDIA 매출 최근 12개월 {pct(nv.rev_yoy*100)}"]
    if tw is not None and len(tw):
        tsmc = tw[tw.ticker == "2330"].sort_values("month").iloc[-1]
        lines.append(f"- TSMC 월매출 {pct(tsmc.yoy)} ({tsmc.month.strftime('%Y-%m')}, 전년 동월 대비)")
    lines.append("- 투자 내역(서버·건물·칩 등 구성)은 공시에 세부가 없어 알 수 없고, 수치는 회사 전체 CAPEX입니다.")
    st.markdown("\n".join(lines))

# ════════════════════════ 하단 ════════════════════════
st.divider()
a, b = st.columns(2, gap="medium")
with a:
    st.markdown("**AI 수익성 경고·주의 신호**")
    watch = [f for f in flags if level_kind(f["level"]) in ("bad", "warn")]
    for f in watch:
        st.markdown(badge(f["level"], level_kind(f["level"])) + f"**{f['title']}**", unsafe_allow_html=True)
    if not watch:
        st.write("현재 경고·주의 신호 없음")
with b:
    st.markdown("**다음에 확인될 것**")
    st.markdown("- 미국 분기 실적 공시(10월 말): CAPEX 증가율과 매출 증가율의 격차, 잉여현금흐름\n"
                "- 대만 AI 공급망 월매출(매월 10일경): 증가율이 20% 아래로 내려가는 회사가 늘어나는지\n"
                "- 빅테크 신규 채권 발행 가산금리: 직전 발행보다 더 넓어지는지")
st.markdown('<p class="note">상단 메뉴: <b>월가 자금</b>(지금·과거·앞으로), <b>AI 수익성</b>(변화·시나리오), <b>갱신 현황</b>(자동 갱신 상태), <b>근거·방법</b>(무엇이 검증됐고 아닌지).</p>', unsafe_allow_html=True)
