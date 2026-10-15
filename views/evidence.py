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

T = load("views/evidence_tests.parquet")
cond = load("views/evidence_conditions.parquet")
k2 = load("views/korea_model_v2.parquet")
P = T[T.primary].set_index("claim")

st.header("1. 무엇이 확인됐나")
c1, c2, c3, c4 = st.columns(4, gap="medium")
k = ev["korea_eval"]
with c1:
    st.markdown(badge("검증됨", "good"), unsafe_allow_html=True)
    st.markdown(f"**미국 마감 → 다음 한국 거래일**\n\n- 시차 약 1거래일, 3일째부터 소멸\n- 모델 상관: 적합 구간 {k['train']['corr']:.2f}/{k['val']['corr']:.2f}, **표본외 {k['test']['corr']:.2f}** (직전 외국인 수급 추가로 0.33에서 상승)\n"
                "- 효과는 거의 시가 갭에서 나옴\n\n"
                f"**팩터 모멘텀**(가격): 전체 t {P.loc['factor_momentum'].all_t:+.1f}, 세 구간 모두 +\n\n"
                f"**산업 모멘텀**(가격): 전체 t {P.loc['industry_momentum'].all_t:+.1f}, 단 {P.loc['industry_momentum'].recent or '최근 구간 확인 필요'}")
with c2:
    st.markdown(badge("단서", "warn"), unsafe_allow_html=True)
    st.markdown(f"**미국 섹터 ETF 자금이 몰리면 이후 덜 오른다**\n\n- 전체 t {P.loc['us_flow_contrarian'].all_t:+.1f}, 2005–18과 2023~은 역방향, 2019–22만 반대\n"
                "- 평온한 시장(VIX 순위 <0.5)에서는 세 구간 모두 역방향(탐색적 분해)\n- VIX가 클수록 이후 수익이 좋은 경향(가설과 반대)\n\n"
                f"전향 장부로 재검증 중({ev['ledger'].get('n_days', 0)}일째)")
with c3:
    st.markdown(badge("참고 · 지지되지 않음", "bad"), unsafe_allow_html=True)
    st.markdown(f"**매크로 국면 → 섹터·자산 수익**: 검정 {ev['regime_testable']:,}건 중 세 구간 재현 {ev['regime_replicated']}건({ev['regime_rate']*100:.1f}%, 우연 기대 약 1.25%)\n\n"
                f"**한국 종목 외국인 순매수 지속**: 28종목 패널에서 어느 호라이즌도 확인 안 됨(전체 t {P.loc['kr_foreign_persistence'].all_t:+.1f})\n\n"
                f"**미국 섹터 ETF 가격 모멘텀**(2005~): 전체 t {P.loc['us_sector_momentum'].all_t:+.1f}\n\n"
                "**AI 투자 → 이후 매출·이익**: 시간 추세를 제거하면 근거 없음")
with c4:
    st.markdown(badge("측정 불가", "mid"), unsafe_allow_html=True)
    st.markdown("**무료 데이터가 없는 것**\n\n- iShares·VanEck 계열 ETF의 실제 자금유입\n- 스타일 팩터의 자금유입(가격 추세로만 확인)\n- 옵션 포지셔닝, 13F, ISM\n- CDS 시세·유통시장 채권 수익률(대용: 공시 기반 신용 지표·주가 기반 Merton·등급대별 OAS), HBM 가격")

st.header("2. 등급을 어떻게 정했나 (같은 규칙으로 전부 다시 계산)")
st.markdown("**규칙**: 세 구간에서 부호가 같고 2구간 이상 |t|≥2이며 전체 |t|≥3이면 **검증됨**, 전체 |t|≥2이고 2구간 이상 같은 부호면 **단서**, 나머지는 **참고**. "
            "t는 Newey-West(겹치는 수익률 길이만큼), 횡단면 신호는 매 시점 스피어만 IC의 시간 평균입니다. 구간 끝의 미래 윈도우는 버립니다(purge).")
st.markdown(ui.grade_table(T, only_primary=False), unsafe_allow_html=True)
st.markdown('<p class="note">같은 주장을 여러 호라이즌으로 친 줄은 <b>대표 하나(사전 지정)만 등급에 쓰는</b> 표시를 둡니다. 보조 줄은 일관성 확인용이고, '
            '예를 들어 미국 ETF 자금은 5일 후에서는 기계적으로 검증됨 기준을 넘지만 2019–22 효과가 거의 0이고 호라이즌을 사후에 고른 것이라 등급에 쓰지 않았습니다.</p>',
            unsafe_allow_html=True)
with st.expander("미국 ETF 자금 역방향은 어떤 조건에서 맞나 (탐색적 분해 · 20일 후)"):
    cc = cond.copy()
    st.dataframe(pd.DataFrame({"조건": cc.condition, "2005–18 IC": cc.train_mean, "t": cc.train_t, "2019–22 IC": cc.val_mean, "t ": cc.val_t,
                               "2023~ IC": cc.test_mean, "t  ": cc.test_t, "전체 t": cc.all_t}).round(2), hide_index=True, width="stretch")
    st.markdown('<p class="note">조건 6개를 사후에 나눠 본 탐색입니다(다중 비교). 가장 일관된 것은 VIX 순위가 낮은 구간(세 구간 모두 음수)이며, 가설 H5로 전향 장부에 등록했습니다.</p>',
                unsafe_allow_html=True)
with st.expander("한국 전달 모델 입력 후보 비교 (train 적합 → val로 선택 → test 한 번)"):
    st.dataframe(pd.DataFrame({"입력 세트": k2.set, "검증 상관": k2.val_corr, "검증 설명력": k2.val_r2, "표본외 상관": k2.test_corr, "표본외 설명력": k2.test_r2}).round(3),
                 hide_index=True, width="stretch")
    st.markdown('<p class="note">후보 4세트를 미리 정하고 검증 구간에서 가장 좋은 것(C: 직전 한국 외국인 수급 추가)을 골랐습니다. 외국인 수급이 하루 단위로 이어지는 성질을 반영한 것이며, '
                '원/달러·HYG·VIX를 더한 D는 개선이 없었습니다. 표본외 설명력은 약 20%로 올랐으나 여전히 방향 힌트 수준입니다.</p>', unsafe_allow_html=True)

st.header("3. AI 수익성 모듈의 신뢰도")
st.markdown("""
- **재무 수치**: SEC 공시 원문(공시일 이후에만 사용). 단, 회사 전체 재무라 AI 전용이 아니며(신뢰 등급 C), 수주잔고·클라우드 부문 매출은 B, 경영진이 직접 공개한 AI 수치는 A입니다.
- **클라우드 부문**: 8-K 실적 발표문의 표를 정규식으로 읽었습니다(AWS·Microsoft 24분기, Google Cloud 13분기 전부 성공). 발표문 형식이 바뀌면 깨질 수 있어 실패 분기는 값을 만들지 않습니다.
- **예측력**: Big4 CAPEX 가속·감속 이후 반도체 주가를 구분하지 못했습니다(독립 사례 4~19개, 신뢰구간 대부분 0 포함). 그래서 종합 점수 대신 상태 신호 10개와 시나리오 도구를 제공합니다.
- **채권 조달**: 신규 발행 시점의 가산금리만 보입니다(CDS·유통시장 수익률은 무료 데이터 없음). 확대는 발행량 증가 때문일 수도 있습니다.
- **안전성(CDS 대용)**: 순부채·이자보상·런웨이·만기 사다리는 SEC 공시 값이라 정확하지만 **상태 지표**이고 예측력은 검증하지 못했습니다(부도·급락 사건이 너무 적음). Merton 내재 스프레드는 실제 발행 조건과 비교해 수준이 맞지 않아(Oracle은 낮게, CoreWeave는 2.3배 높게) 방향만 참고합니다. 신용등급대별 OAS는 ICE BofA(FRED, 최근 3년)입니다. 증분 ROIC는 영업이익 증가를 전부 추가 투자의 성과로 보는 낙관적 계산입니다.
- **금융 조건의 전이**: 차입 의존 기업(Oracle·CoreWeave)의 주가 스트레스와 HY·CCC 스프레드 확대는 이후 20일 NVIDIA·반도체 수익을 앞서지 못했습니다(|t|<1.5, 같은 날 동행 상관만 0.4~0.5). 자금 조달 시나리오는 산식이지 예측이 아닙니다.
""")
if aiev is not None:
    e = aiev[(aiev.split == "all") & (aiev.h == 20) & aiev.asset.isin(["SOXX", "NVDA", "000660.KS"]) &
             aiev.cond.isin(["capex_accelerating", "capex_decelerating", "fcf_rising", "fcf_falling", "revenue_accelerating"])]
    with st.expander("이벤트 스터디 표 (Big4 실적 공시일 이후 20일 수익률)"):
        st.dataframe(pd.DataFrame({"조건": e.cond, "자산": e.asset, "독립 사례": e.n_eff, "이후 20일 평균%": e["mean"] * 100, "평소 대비 %p": e.excess * 100,
                                   "신뢰구간 하단": e.ci_lo * 100, "신뢰구간 상단": e.ci_hi * 100}).round(1), hide_index=True, width="stretch")

_fe = load("views/ai_forecast_eval.parquet")
if _fe is not None:
    with st.expander("AI 다음 12개월 기준선 전망: 방법별 과거 정확도 (Big4 분기 합산)"):
        _t = _fe[_fe.period.isin(["2016–19", "2020–22", "2023~", "전체"])].copy()
        _t["값"] = _t.apply(lambda r: f"{r.bias*100:+.0f}% / {r.rmse*100:.0f}%", axis=1)
        _pv = _t.pivot_table(index=["metric", "method"], columns="period", values="값", aggfunc="first")[["2016–19", "2020–22", "2023~", "전체"]].reset_index()
        _pv["metric"] = _pv.metric.map({"capex": "CAPEX", "ocf": "영업현금흐름", "revenue": "매출", "op_income": "영업이익", "fcf": "FCF(영업현금흐름 대비 %p)"})
        st.dataframe(_pv.rename(columns={"metric": "지표", "method": "방법"}), hide_index=True, width="stretch")
        st.markdown('<p class="note">각 칸은 <b>편향 / RMSE</b>(다음 4분기 합계의 로그 오차). 0에 가까운 편향이 좋습니다. "최근 분기 YoY 지속"이 매출·영업현금흐름·영업이익에서 가장 덜 치우쳤고, '
                    'CAPEX는 2023년 이후 15% 낮게 맞혔습니다. 방법을 5개 중 최선으로 골랐고 4분기 창이 겹쳐 독립 표본이 약 9개라 등급은 단서입니다.</p>', unsafe_allow_html=True)

st.header("4. 전향 검증 장부")
L = ev.get("ledger", {})
if L.get("n_days"):
    st.write(f"기록 시작 {L['started']} · {L['n_days']}일. 과거를 되메워 넣지 않고, 등록한 가설을 앞으로의 실제 데이터로만 판정합니다(규칙: docs/FORWARD_TEST.md).")
    rows = [("H1 미국→한국 다음 장 외국인 수급", L["h1_korea"]["n"], "n≥60에서 상관 0.2 유지"),
            ("H2 공포(VIX 상위) 뒤 SPY 20일 수익이 높다", L["h2_vix"]["n_high_fear"], "독립 사례 8건 이상"),
            ("H3 한국 업종 외국인 60일 흐름 → 20일 상대수익", L["h3_kr_sector"]["n_matured"], "비겹침 12개, IC 0.05 이상"),
            ("H4 미국 섹터 ETF 순유입이 클수록 이후 상대수익이 낮다", L.get("h4_us_flow", {}).get("n_matured", 0), "비겹침 12개, IC −0.05 이하"),
            ("H5 H4가 평온한 시장(VIX 순위<0.5)에서 더 뚜렷하다", L.get("h5_us_flow_calm", {}).get("n_matured", 0), "비겹침 12개, IC −0.05 이하"),
            ("H6 팩터 모멘텀(5팩터 평균) 월 수익이 +", L.get("h6_factor_mom", {}).get("n_months", 0), "새로 발표된 월 12개 이상, 평균 > 0"),
            ("H7 산업 모멘텀 IC가 +", L.get("h7_industry_mom", {}).get("n_months", 0), "새로 발표된 월 12개 이상, 평균 IC > 0")]
    st.dataframe(pd.DataFrame(rows, columns=["가설", "실현된 표본", "판정 조건"]), hide_index=True, width="stretch")

st.header("5. 데이터 출처와 갱신")
st.markdown("갱신 일정, 소스별 신선도, 최근 실행 이력은 **갱신 현황** 페이지에서 봅니다.")
st.markdown('<p class="note">전체 방법론·규칙: GitHub gtjdbfla/macro-flow-dashboard 의 docs/DESIGN · FEATURES · BACKTEST · RESULTS_SUMMARY · AI_PROFITABILITY · FORWARD_TEST.</p>',
            unsafe_allow_html=True)
