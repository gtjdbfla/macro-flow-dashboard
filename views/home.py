"""홈: 맨 위에 두 질문의 답(요약 카드 한 쌍) → 아래에서 한쪽만 골라 자세히 본다.
스크롤을 줄이고(한 번에 한 주제), 글 대신 큰 숫자·표·칩으로 보여준다."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import badge, callout, chips, duo, ev_card, kpis, level_kind, load, pct, plot, summary_card, trio

meta, state = load("views/meta.json"), load("views/state.json")
sec = load("results/sector_snapshot.parquet")
kh = load("views/korea_model_recent.parquet")
quint = load("views/korea_quintile.parquet")
latest = load("results/ai_latest.parquet")
agg = load("results/ai_cloud_agg.parquet")
usd = load("views/sector_flow_usd.parquet")
nxt = load("views/flow_next.json")
ha = load("views/home_ai.parquet")
cq = load("views/ai_capex_q.parquet")
ev = load("views/evidence.json")
tw = ui.load_parquet_dir("twse_monthly.parquet")
flags = json.loads(open(ui.DATA_DIR / "results/ai_flags.json", encoding="utf-8").read())["flags"] if (ui.DATA_DIR / "results/ai_flags.json").exists() else []

# ── 머리말 ───────────────────────────────────────────────────────────────────
st.title("Macro & Capital Flow")
st.markdown('<p class="lede">① 월가의 자금이 어디에 있고 어떻게 움직여 왔는지, ② AI 기업의 수익성이 어떻게 변하고 있는지를 데이터로 봅니다. '
            '앞날을 맞히는 도구가 아니라, 확인된 것과 아직 확인되지 않은 것을 구분해 보여주는 현황판입니다.</p>', unsafe_allow_html=True)
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
press = [i for i in ("F1", "F2", "F3", "F9", "F10") if i in fl]
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
head2 = f"{verdict} 수요 신호 {n_good}/{len(demand)}개 긍정, 회수·조달 신호 {n_bad}/{len(press)}개 경고·주의."
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
    st.markdown("### 자금은 어디로 들어오고 어디서 빠졌나 (최근 20일, 실제 ETF 설정·환매)")
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

    st.markdown("### 과거 백테스트로 본 '다음에 자금이 몰릴 곳' 후보")
    rules = pd.DataFrame(nxt["rules"])
    u = rules[rules.rule == "us_flow"].set_index("split")
    k = rules[rules.rule == "kr_foreign"].set_index("split")
    tag = lambda df, s: (f"{df.loc[s].mean_spread:+.2f}%p", f"t {df.loc[s].t:+.1f}") if s in df.index else ("—", "")  # noqa: E731
    ua = u.loc["all"]
    mark = lambda df, s, sign: "" if s not in df.index else ("✓ 같은 방향" if df.loc[s].mean_spread * sign > 0 else "✗ 반대")  # noqa: E731
    card1 = ev_card(badge("단서 · 약함", "warn"), "자금이 몰린 섹터는 이후 20일 덜 오르는 경향", f"{ua.hit:.0f}%",
                    f"하위 3섹터가 상위 3섹터보다 더 오른 비율 (비겹침 {int(ua.n)}번, 평균 {ua.mean_spread:+.2f}%p, t {ua.t:+.1f})",
                    [("2005–18", " ".join(tag(u, "train")), mark(u, "train", -1)), ("2019–22", " ".join(tag(u, "val")), mark(u, "val", -1)),
                     ("2023~", " ".join(tag(u, "test")), mark(u, "test", -1))],
                    chips("지금 부진 가능(자금 쏠림)", [n for n, _ in nxt["in_top"]], "dn") + chips("지금 유리 가능(자금 이탈)", [n for n, _ in nxt["out_top"]], "up"))
    ka = k.loc["all"]
    card2 = ev_card(badge("참고 · 유의하지 않음", "mid"), "한국: 외국인이 산 업종은 이후에도 약간 우위", f"{ka.mean_spread:+.2f}%p",
                    f"외국인 60일 순매수 비중 상위3 − 하위3, 이후 20일 KOSPI 대비 (t {ka.t:+.1f}, {int(ka.n)}번)",
                    [("2019–22", " ".join(tag(k, "val")), mark(k, "val", 1)), ("2023~", " ".join(tag(k, "test")), mark(k, "test", 1)), ("전체", " ".join(tag(k, "all")), "")],
                    chips("현재 상위", [n for n, _ in nxt["kr_in_top"]], "up") + chips("현재 하위", [n for n, _ in nxt["kr_out_top"]], "dn"))
    q = quint[quint.period == "oos"]
    card3 = ev_card(badge("검증됨", "good"), "미국 마감 → 다음 한국 거래일 외국인 수급", f"{z:+.2f}", f"모델값(z) · 지금은 {zdir} (표본외 상관 {ev['korea_eval']['test']['corr']:.2f})",
                    [("예측 최상위 구간", "순매수 확률", f"{q.buy_prob.iloc[-1]:.0f}%"), ("예측 최하위 구간", "순매수 확률", f"{q.buy_prob.iloc[0]:.0f}%"),
                     ("평소", "순매수 확률", "약 47%")],
                    '<div class="chipl">효과는 거의 시가 갭에서 나오고 개장 후 초과수익은 확인되지 않았습니다.</div>')
    trio(card1, card2, card3)
    callout("이 후보들은 <b>예측이 아니라 약한 과거 패턴을 지금 값에 적용한 것</b>입니다. 첫 번째 카드의 적중 비율은 동전 던지기보다 조금 나은 수준이고 한 구간에서는 반대였습니다. "
            "매크로 국면이나 상대강도로 섹터를 맞히는 것은 백테스트에서 우연 수준이었고, 스타일 팩터(모멘텀·가치·퀄리티 등)는 무료 자금유입 데이터가 없어 아직 검증하지 못했습니다.")


def ai_detail():
    st.markdown("### 누가 투자하나 — 회사별 12개월 CAPEX와 변화")
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
                f"**가장 빠르게 늘리는 곳**: {t.sort_values('capex_yoy', ascending=False).iloc[0]['name']} (+{t.capex_yoy.max()*100:.0f}%)\n"
                f"- **영업현금흐름 대비 부담이 2분기 전보다 커진 곳**: {', '.join(worse) or '없음'} · **줄어든 곳**: {', '.join(better) or '없음'}")

    st.markdown("### 변화량 — 1년 전보다 얼마를 더 쓰고, 얼마를 더 벌었나")
    a, b = st.columns(2, gap="large")
    with a:
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
        plot(fig, 330)
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

    st.markdown("### 투자를 받는 쪽")
    nv = latest[latest.ticker == "NVDA"].iloc[0]
    tiles = [("NVIDIA 매출", pct(nv.rev_yoy * 100), "최근 12개월, 전년 대비")]
    if tw is not None and len(tw):
        tsmc = tw[tw.ticker == "2330"].sort_values("month").iloc[-1]
        tiles.append(("TSMC 월매출", pct(tsmc.yoy), f"{tsmc.month.strftime('%Y-%m')}, 전년 동월 대비"))
    tiles.append(("클라우드 3사 매출", pct(cl.yoy * 100), "최근 분기, 전년 동기 대비"))
    st.markdown(kpis(tiles), unsafe_allow_html=True)
    callout("투자 내역(서버·건물·칩 등 구성)은 공시에 세부가 없어 알 수 없고, 위 CAPEX는 <b>회사 전체</b> 수치입니다. 이 페이지는 현재 상태와 방향을 보여줄 뿐 주가를 예측하지 않습니다.", "info")


detail()

# ── 하단 ─────────────────────────────────────────────────────────────────────
st.divider()
a, b = st.columns(2, gap="large")
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
                "- 대만 AI 공급망 월매출(매월 10일경): 증가율 20% 아래 회사가 늘어나는지\n"
                "- 빅테크 신규 채권 발행 가산금리: 직전보다 더 넓어지는지")
st.markdown('<p class="note">상단 메뉴: <b>월가 자금</b>(지금·과거·앞으로), <b>AI 수익성</b>(변화·시나리오), <b>갱신 현황</b>, <b>근거·방법</b>.</p>', unsafe_allow_html=True)
