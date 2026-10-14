"""홈: 두 질문의 답을 한 화면에."""
import json

import pandas as pd
import streamlit as st

import ui
from ui import badge, card, level_kind, load, pct

meta, state = load("views/meta.json"), load("views/state.json")
S = load("results/summary.json")
sec = load("results/sector_snapshot.parquet")
cf = load("views/cftc_now.parquet")
kh = load("views/korea_model_recent.parquet")
latest = load("results/ai_latest.parquet")
agg = load("results/ai_cloud_agg.parquet")
flags = json.loads(open(ui.DATA_DIR / "results/ai_flags.json", encoding="utf-8").read())["flags"] if (ui.DATA_DIR / "results/ai_flags.json").exists() else []

st.title("Macro & Capital Flow")
st.markdown('<p class="lede">월가의 자금이 어디에 있고 어디로 움직여 왔는지, 그리고 AI 기업의 수익성이 어떻게 변하고 있는지를 데이터로 봅니다. '
            '앞날을 맞히는 도구가 아니라, 확인된 것과 아직 확인되지 않은 것을 구분해서 보여주는 현황판입니다.</p>', unsafe_allow_html=True)
ui.freshness(meta)
st.caption(f"시장 상태: {state['regime']} 국면(성장 {'가속' if state['growth']=='Accelerating' else '둔화'}, 물가 "
           f"{'가속' if state['inflation']=='Accelerating' else '둔화'}) · 정책 {state['policy']} · 유동성 {state['liquidity']} · 달러 {state['dollar']}"
           f" — {state['days_in_regime']}거래일째 유지")

# ── ① 월가 자금 ────────────────────────────────────────────────────────────
etf = sec[sec.flow_src == "ETF유입"].copy()
top, bot = etf.sort_values("flow_1m", ascending=False).iloc[0], etf.sort_values("flow_1m", ascending=True).iloc[0]
sp_row = cf[cf.name == "S&P 500(전체)"].iloc[0]
last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
z = float(last.pred_foreign_z)
zdir = "순매수 쪽" if z > 0.15 else ("순매도 쪽" if z < -0.15 else "중립")
head1 = (f"최근 20일 ETF 자금은 <b>{top['name']}</b>({top.flow_1m*100:+.1f}%)로 가장 많이 들어오고 "
         f"<b>{bot['name']}</b>({bot.flow_1m*100:+.1f}%)에서 가장 많이 나갔습니다(순자산 대비). "
         f"미국 마감 기준으로 다음 한국 장의 외국인은 <b>{zdir}</b>일 가능성이 높습니다.")
c1, c2 = st.columns(2, gap="medium")
with c1:
    card("① 월가의 자금은 어디에 있나", badge("한국 전달만 검증됨", "info") + badge("섹터 흐름은 현황 파악용", "mid"), head1,
         [("순유입 1위 섹터(20일)", top["name"], f"순자산의 {top.flow_1m*100:+.1f}%"),
          ("한국 다음 장 외국인 수급(z)", f"{z:+.2f}", zdir),
          ("S&P500 선물 자산운용 포지션", f"{sp_row.am_rank:.0f}분위", "과거 이력 대비 위치")])

# ── ② AI 수익성 ────────────────────────────────────────────────────────────
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
with c2:
    card("② AI 수익성은 어떻게 변하고 있나", badge("현황판(예측 아님)", "warn"), head2,
         [("Big4 투자 vs 매출 증가", f"{pct(B.capex_yoy*100)} / {pct(B.rev_yoy*100)}", "CAPEX / 매출, 최근 12개월"),
          ("클라우드 3사 매출·마진", f"{pct(cl.yoy*100)} · {cl.margin*100:.0f}%", f"1년 전 마진 {cl.margin_prior*100:.0f}%"),
          ("잉여현금흐름", f"{B.fcf_ttm/1e8:,.0f}억 달러", f"전년 대비 {B.fcf_chg/1e8:+,.0f}억 달러")])

# ── 지금 주의 깊게 볼 것 ───────────────────────────────────────────────────
st.markdown("### 지금 눈여겨볼 것")
watch = [f for f in flags if level_kind(f["level"]) in ("bad", "warn")]
a, b = st.columns(2, gap="medium")
with a:
    st.markdown("**AI 수익성 경고·주의 신호**")
    for f in watch:
        st.markdown(badge(f["level"], level_kind(f["level"])) + f"**{f['title']}**", unsafe_allow_html=True)
    if not watch:
        st.write("현재 경고·주의 신호 없음")
with b:
    st.markdown("**다음에 확인될 것**")
    st.markdown("- 미국 분기 실적 공시(10월 말): CAPEX 증가율과 매출 증가율의 격차, 잉여현금흐름\n"
                "- 대만 AI 공급망 월매출(매월 10일경): 증가율이 20% 아래로 내려가는 회사가 늘어나는지\n"
                "- 빅테크 신규 채권 발행 가산금리: 직전 발행보다 더 넓어지는지")
st.markdown('<p class="note">페이지 상단 메뉴에서 <b>월가 자금</b>(지금·과거·앞으로), <b>AI 수익성</b>(변화·시나리오), <b>근거·방법</b>(무엇이 검증됐고 무엇이 아닌지)으로 이동합니다.</p>',
            unsafe_allow_html=True)
