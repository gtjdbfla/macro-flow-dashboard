"""'돈의 흐름 브리핑' — 프로젝트의 다섯 질문에 대한 자동 요약. 사실(계산값)과 근거 등급만 쓰고 예측 문장은 쓰지 않는다.
등급: 검증=표본외 확인, 참고=예측력 미확인(현재 위치를 보여줄 뿐)."""
import json

import numpy as np
import pandas as pd

from src.data import DATA_DIR, PARQUET

R = DATA_DIR / "results"
CFTC_SECTOR = {"Technology": "SP_Tech_E-mini", "Financials": "SP_Fin_E-mini", "Energy": "SP_Energy_E-mini",
               "Healthcare": "SP_Health_E-mini", "Industrials": "SP_Ind_E-mini", "Utilities": "SP_Util_E-mini",
               "ConsDisc": "SP_ConsDisc_E-mini", "ConsStaples": "SP_ConsStaples_E-mini",
               "Materials": "SP_Materials_E-mini"}
KR_NAMES = {"Semis": "반도체", "Auto": "자동차", "Banks": "은행", "Healthcare": "헬스케어", "Defense": "방산",
            "Shipbuilding": "조선", "EnergyChem": "에너지화학", "Battery": "2차전지"}


def _names(df, n, col="score", top=True):
    d = df.sort_values(col, ascending=not top).head(n)
    return ", ".join(f"{r['name']}({r['direction']})" for _, r in d.iterrows())


def build(summary: dict) -> list:
    out = []
    st = pd.read_parquet(PARQUET / "features_state.parquet").dropna(subset=["state_key"]).iloc[-1]
    sc = summary["scores"]
    out.append({"q": "① 지금 매크로·유동성·월가 환경은?", "grade": "참고",
                "text": f"{st['regime']} 국면(성장 {'가속' if st['growth']=='Accelerating' else '둔화'} · 물가 "
                        f"{'가속' if st['inflation']=='Accelerating' else '둔화'}), 정책 {st['policy']}, 유동성 {st['liquidity']}, "
                        f"달러 {st['dollar']}. 위험선호 {sc['risk']:.0f}/100, 기관 포지셔닝 {sc['flow']:.0f}/100. "
                        "※ 이 점수들은 예측력이 검증되지 않은 현재 위치 요약입니다."})
    sec = pd.read_parquet(R / "sector_snapshot.parquet")
    if len(sec):
        s = sec.copy()
        enter = s[(s.rs_1m < 0) & (s.rs_1w > 0) & (s.rs_3d > 0) & (s.cmf_1w > 0)].sort_values("cmf_1w", ascending=False).head(3)
        leave = s[(s.rs_1m > 0) & (s.rs_1w < 0) & (s.rs_3d < 0) & (s.cmf_1w < 0)].sort_values("cmf_1w").head(3)
        etf = s[s.flow_src == "ETF유입"]
        fl = ""
        if len(etf) >= 6:
            top = etf.sort_values("flow_1m", ascending=False).head(2)
            bot = etf.sort_values("flow_1m").head(2)
            tops = ", ".join("%s(%+.1f%%)" % (r["name"], r["flow_1m"] * 100) for _, r in top.iterrows())
            bots = ", ".join("%s(%+.1f%%)" % (r["name"], r["flow_1m"] * 100) for _, r in bot.iterrows())
            fl = f" 실제 ETF 순유입(20일, %AUM) 큰 섹터: {tops} / 순유출: {bots}."
        txt = (f"가격+자금 합성 상위: {_names(s, 3)}. 하위: {_names(s, 3, top=False)}."
               + (f" 유입 전환 후보(1M 약세→1W·3D 강세, 자금 유입 큰 순 최대 3개): {', '.join(enter['name'])}." if len(enter) else "")
               + (f" 이탈 전환 후보(1M 강세→1W·3D 약세, 자금 유출 큰 순 최대 3개): {', '.join(leave['name'])}." if len(leave) else ""))
        out.append({"q": "② 월가의 돈은 어느 섹터로 움직이나?", "grade": "참고",
                    "text": txt + fl + " ※ 실제 설정·환매 기반이지만(SPDR ETF), 이후 수익률 예측력은 약한 역방향 경향만 보여 검증 중입니다."})
    fw = pd.read_parquet(R / "analog_forward.parquet")
    A = summary["analog"]
    spy = fw[(fw.asset == "SPY") & (fw.kind == "abs") & (fw.h == 20)]
    rel = fw[(fw.kind == "rel") & (fw.h == 20) & (fw.n_eff >= 8) & fw.asset.isin(
        ["XLE", "XLF", "XLK", "XLU", "XLV", "XLI", "XLP", "XLY", "XLB", "XLC", "XLRE"])].copy()
    t3 = ""
    if len(rel):
        rel["ex"] = rel["median"] - rel["base_median"]
        top, bot = rel.sort_values("ex", ascending=False).head(2), rel.sort_values("ex").head(2)
        t3 = (f" 20일 SPY 대비 중앙값이 무조건 평균보다 높았던 섹터: {', '.join(f'{a}({e*100:+.1f}%p)' for a, e in zip(top.asset, top.ex))}"
              f" / 낮았던 섹터: {', '.join(f'{a}({e*100:+.1f}%p)' for a, e in zip(bot.asset, bot.ex))}.")
    sp = ""
    if len(spy):
        r = spy.iloc[0]
        sp = f" SPY 20일 중앙값 {r['median']*100:+.1f}% (무조건 {r.base_median*100:+.1f}%), 승률 {r.win*100:.0f}% (무조건 {r.base_win*100:.0f}%)."
    rep = pd.read_parquet(R / "regime_replication.parquet")
    rep = rep[rep.testable]
    rrate = f"{rep.replicated.mean()*100:.1f}%"
    out.append({"q": "③ 역사적으로 비슷한 환경 뒤에 돈은 어디로 갔나?", "grade": "참고",
                "text": f"유사 국면 {A['n_episodes']}사례({A['n_days']}일).{sp}{t3} ※ 레짐→섹터 매핑은 3개 구간 재현율이 {rrate}로 우연 수준(≈1.25%)이라 "
                        "약한 단서로만 보세요."})
    km = summary["korea_model"]
    kh = pd.read_parquet(R / "korea_model_history.parquet")
    last = kh.dropna(subset=["pred_foreign_z"]).iloc[-1]
    K = pd.read_parquet(PARQUET / "features_korea.parquet").dropna(how="all").iloc[-1]
    rows = [(KR_NAMES[s], K.get(f"kr__{s}_foreign__ratio_20d", np.nan)) for s in KR_NAMES]
    rows = sorted([r for r in rows if r[1] == r[1]], key=lambda x: -x[1])
    ktxt = ""
    if rows:
        buy = [f"{n}({v*100:+.1f}%)" for n, v in rows[:2]]
        sell = [f"{n}({v*100:+.1f}%)" for n, v in rows[-2:][::-1]]
        ktxt = (f" 최근 20거래일 업종별 외국인 순매수/거래대금: 가장 강한 쪽 {', '.join(buy)}, "
                f"가장 약한 쪽 {', '.join(sell)}" + (" (전 업종 순매도 — 상대적 순위)." if rows[0][1] < 0 else "."))
    out.append({"q": "④ 그 흐름이 한국에는 며칠 뒤 어떻게 전달되나?", "grade": "검증",
                "text": f"미국 마감 → 다음 한국 거래일(시차 ≈1일)에 외국인 수급 z {last.pred_foreign_z:+.2f} 예상"
                        f"(표본외 상관 {km['eval']['test']['corr']:.2f}).{ktxt}"})
    res = (kh.actual_next_foreign_z - kh.pred_foreign_z).dropna()
    r20 = res.tail(20).mean() / res.loc[:"2022-12-31"].std()
    verdict = ("미국 신호보다 외국인이 **더 약하게** 반응(매도 우위 쪽 괴리)" if r20 < -0.5 else
               "미국 신호보다 외국인이 **더 강하게** 반응(매수 우위 쪽 괴리)" if r20 > 0.5 else "역사적 전달 패턴을 대체로 따름")
    out.append({"q": "⑤ 한국은 그 패턴을 따라가나, 괴리가 있나?", "grade": "검증",
                "text": f"최근 20거래일 평균 잔차 {r20:+.2f}σ → {verdict}."})
    return out
