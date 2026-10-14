"""근거·방법: 무엇이 검증됐고 무엇이 아닌지, 데이터는 어디서 왔고 얼마나 신선한지."""
import pandas as pd
import streamlit as st

import ui
from ui import badge, load

ev = load("views/evidence.json")
meta = load("views/meta.json")
quint = load("views/korea_quintile.parquet")
cov = load("coverage.json")
aiev = load("results/ai_event_study.parquet")

st.title("근거·방법")
st.markdown('<p class="lede">이 대시보드의 모든 숫자는 공개 데이터로 계산했고, 과거에서 찾은 관계는 <b>세 개 구간(2005-18 · 2019-22 · 2023~)에서 같은 방향으로 재현되는지</b>로 걸렀습니다. '
            '아래는 그 결과입니다.</p>', unsafe_allow_html=True)
ui.freshness(meta)

st.header("1. 무엇이 확인됐나")
c1, c2, c3, c4 = st.columns(4, gap="medium")
k = ev["korea_eval"]
with c1:
    st.markdown(badge("검증됨", "good"), unsafe_allow_html=True)
    st.markdown(f"**미국 마감 → 다음 한국 거래일**\n\n- 시차 약 1거래일, 3일째부터 소멸\n- 모델 상관: 적합 구간 {k['train']['corr']:.2f}/{k['val']['corr']:.2f}, **표본외 {k['test']['corr']:.2f}**\n"
                "- 미국 risk-on 뒤 외국인 순매수 확률 62~64%(평소 47~50%)\n- 효과는 거의 시가 갭에서 나옴")
with c2:
    st.markdown(badge("단서", "warn"), unsafe_allow_html=True)
    st.markdown("**약하지만 방향이 일관**\n\n- 한국 업종 외국인 60일 흐름 → 이후 20일 업종 수익(상관 약 0.07)\n- 미국 ETF 순유입 → 이후 상대수익은 낮은 경향\n- VIX가 클수록 이후 수익이 좋은 경향(가설과 반대)\n\n"
                f"모두 전향 장부로 재검증 중({ev['ledger'].get('n_days', 0)}일째)")
with c3:
    st.markdown(badge("지지되지 않음", "bad"), unsafe_allow_html=True)
    st.markdown(f"**매크로 국면 → 섹터·자산 수익**\n\n- 검정 {ev['regime_testable']:,}건 중 세 구간 재현 {ev['regime_replicated']}건({ev['regime_rate']*100:.1f}%, 우연 기대 약 1.25%)\n"
                f"- 섹터 상대강도·모멘텀·자금흐름의 예측력은 구간마다 부호가 바뀜\n- 점수 구성요소 {ev['score_components']}개 중 채택 {ev['score_included']}개")
with c4:
    st.markdown(badge("측정 불가", "mid"), unsafe_allow_html=True)
    st.markdown("**무료 데이터가 없는 것**\n\n- iShares·VanEck 계열 ETF의 실제 자금유입\n- 옵션 포지셔닝, 13F, ISM\n- CDS·유통시장 채권 수익률, HBM 가격\n- 한국 반도체 수출 속보")

st.header("2. 한국 전달 모델: 예측 5분위별 실제 결과")
q = quint.copy()
q["period"] = q["period"].map({"fit": "적합 구간 2005-2022", "oos": "표본외 2023~"})
st.dataframe(q.rename(columns={"period": "구간", "q": "예측 분위(Q1=가장 약세)", "days": "일수", "buy_prob": "외국인 순매수 확률%", "total_bp": "다음날 KOSPI bp",
                               "gap_bp": "그중 시가 갭 bp", "intraday_bp": "장중 bp"}).round(1), hide_index=True, width="stretch")
st.markdown('<p class="note">입력은 QQQ·SOXX·달러지수 일수익률과 실질금리 변화. 상위 분위일수록 외국인 순매수 확률과 다음날 KOSPI가 높지만, 상승분이 시가 갭에 몰려 있어 개장 후에 잡을 수 있는 초과수익은 확인되지 않았습니다.</p>',
            unsafe_allow_html=True)

st.header("3. AI 수익성 모듈의 신뢰도")
st.markdown("""
- **재무 수치**: SEC 공시 원문(공시일 이후에만 사용). 단, 회사 전체 재무라 AI 전용이 아니며(신뢰 등급 C), 수주잔고·클라우드 부문 매출은 B, 경영진이 직접 공개한 AI 수치는 A입니다.
- **클라우드 부문**: 8-K 실적 발표문의 표를 정규식으로 읽었습니다(AWS·Microsoft 24분기, Google Cloud 13분기 전부 성공). 발표문 형식이 바뀌면 깨질 수 있어 실패 분기는 값을 만들지 않습니다.
- **예측력**: Big4 CAPEX 가속·감속 이후 반도체 주가를 구분하지 못했습니다(독립 사례 4~19개, 신뢰구간 대부분 0 포함). 그래서 종합 점수 대신 상태 신호 10개와 시나리오 도구를 제공합니다.
- **채권 조달**: 신규 발행 시점의 가산금리만 보입니다(CDS·유통시장 수익률은 무료 데이터 없음). 확대는 발행량 증가 때문일 수도 있습니다.
""")
if aiev is not None:
    e = aiev[(aiev.split == "all") & (aiev.h == 20) & aiev.asset.isin(["SOXX", "NVDA", "000660.KS"]) &
             aiev.cond.isin(["capex_accelerating", "capex_decelerating", "fcf_rising", "fcf_falling", "revenue_accelerating"])]
    with st.expander("이벤트 스터디 표 (Big4 실적 공시일 이후 20일 수익률)"):
        st.dataframe(pd.DataFrame({"조건": e.cond, "자산": e.asset, "독립 사례": e.n_eff, "이후 20일 평균%": e["mean"] * 100, "평소 대비 %p": e.excess * 100,
                                   "신뢰구간 하단": e.ci_lo * 100, "신뢰구간 상단": e.ci_hi * 100}).round(1), hide_index=True, width="stretch")

st.header("4. 전향 검증 장부")
L = ev.get("ledger", {})
if L.get("n_days"):
    st.write(f"기록 시작 {L['started']} · {L['n_days']}일. 과거를 되메워 넣지 않고, 등록한 가설을 앞으로의 실제 데이터로만 판정합니다(규칙: docs/FORWARD_TEST.md).")
    rows = [("H1 미국→한국 다음 장 외국인 수급", L["h1_korea"]["n"], "n≥60에서 상관 0.2 유지"),
            ("H2 공포(VIX 상위) 뒤 SPY 20일 수익이 높다", L["h2_vix"]["n_high_fear"], "독립 사례 8건 이상"),
            ("H3 한국 업종 외국인 60일 흐름 → 20일 상대수익", L["h3_kr_sector"]["n_matured"], "비겹침 12개, IC 0.05 이상"),
            ("H4 미국 섹터 ETF 순유입이 클수록 이후 상대수익이 낮다", L.get("h4_us_flow", {}).get("n_matured", 0), "비겹침 12개, IC −0.05 이하")]
    st.dataframe(pd.DataFrame(rows, columns=["가설", "실현된 표본", "판정 조건"]), hide_index=True, width="stretch")

st.header("5. 데이터 출처와 갱신")
st.markdown("갱신 일정, 소스별 신선도, 최근 실행 이력은 **갱신 현황** 페이지에서 봅니다.")
st.markdown('<p class="note">전체 방법론·규칙: GitHub gtjdbfla/macro-flow-dashboard 의 docs/DESIGN · FEATURES · BACKTEST · RESULTS_SUMMARY · AI_PROFITABILITY · FORWARD_TEST.</p>',
            unsafe_allow_html=True)
