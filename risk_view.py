"""위험 온도계 카드 — 홈 맨 위. 투자 지시가 아니라 '큰 낙폭이 올 위험이 높은 상태인가'를 보여준다(scoring/risk_gauge.py)."""
import json

import plotly.graph_objects as go
import streamlit as st

import ui
from ui import badge, load, plot

KIND = {"낮음": "good", "보통": "good", "높음": "warn", "매우 높음": "bad"}
COLOR = {"낮음": "#22a55a", "보통": "#8bc34a", "높음": "#e08a1e", "매우 높음": "#ee5555"}
BANDS = [(0.0, 0.4, "낮음"), (0.4, 0.6, "보통"), (0.6, 0.8, "높음"), (0.8, 1.0, "매우 높음")]


def render():
    p = ui.DATA_DIR / "results/risk_gauge.json"
    if not p.exists():
        return
    g = json.loads(p.read_text(encoding="utf-8"))
    lv = load("results/risk_gauge_levels.parquet")
    hist = load("results/risk_gauge_history.parquet")
    kor = load("results/risk_gauge_kospi.parquet")
    row = lv[lv.level == g["level"]].iloc[0] if lv is not None and (lv.level == g["level"]).any() else None
    kind = KIND[g["level"]]
    prob = f"과거 이 구간에서 6개월 안에 -15% 이상 빠진 비율 <b>{row.p_bad*100:.0f}%</b> (전체 평균 {g['prevalence']*100:.0f}%)" if row is not None else ""
    shock = ""
    if g["shock"]:
        ss = g["shock_stats"]
        shock = (f'<div class="callout">⚠ <b>급변 표지</b> — 최근 5일 안에 S&amp;P500 {g["ret5"]*100:+.1f}% · VIX {g["vix5"]*100:+.0f}%. '
                 f'과거 {ss["events"]}번 이런 일이 있었고 그중 {ss["p_bad"]*100:.0f}%가 이후 6개월 내 -15%를 넘게 빠졌습니다(평소 {g["prevalence"]*100:.0f}%). '
                 f'다만 6개월 뒤 수익 중앙값은 {ss["ret6_median"]*100:+.0f}%로 반등도 잦았습니다. 점수는 이 급변을 며칠 늦게 반영합니다.</div>')
    kd = g.get("kospi_dd")
    items = [("S&P500 고점 대비", f"{g['spx_dd']*100:.1f}%", f"{g['spx']:,.0f}"), ("VIX", f"{g['vix']:.1f}", "변동성 지수"),
             ("코스피 고점 대비", "—" if kd is None else f"{kd*100:.1f}%", "" if g.get("kospi_level") is None else f"{g['kospi_level']:,.0f}")]
    st.markdown(f'<div class="sum"><div><span class="t">시장 위험 온도계</span> {badge(g["level"], kind)} '
                f'<span class="muted">{g["as_of"]} 기준 · 점수 {g["score"]*100:.0f}/100</span></div>'
                f'<div class="head">{prob}</div>{ui.kpis(items)}</div>{shock}', unsafe_allow_html=True)
    ui.note(
        "<p class='note'><b>무엇인가</b>: S&P500이 앞으로 6개월 안에 -15% 이상 빠질 위험이 높은 상태인지를 5개 시장 지표(200일선 이탈·12개월 모멘텀·1년 고점 대비 낙폭·변동성·달러 강세)의 "
        "과거 대비 위치로 보여줍니다. 매수·매도 신호가 아닙니다.<br><b>한계</b>: 위험이 높다고 평균 수익이 마이너스는 아닙니다(큰 하락 뒤 큰 반등이 잦음). "
        "1967~2006 데이터로 지표를 골랐고 2007년 이후 구분력(AUC)은 약 0.56으로 약합니다. 독립적인 대폭락 사례가 적어 점수는 거친 지도입니다.</p>"
        + "".join(f"<p class='note'>{r.level}: {int(r.days):,}일 · 큰 낙폭 확률 {r.p_bad*100:.0f}% · 앞 6개월 평균 수익 {r.ret6_mean*100:+.1f}%</p>" for r in lv.itertuples()) if lv is not None else "")
    if hist is not None and len(hist):
        @st.fragment
        def chart():
            rng = st.segmented_control("기간", ["3년", "10년", "전체"], default="3년", key="rg_rng", label_visibility="collapsed")
            h = hist.tail({"3년": 156, "10년": 520}.get(rng or "3년", len(hist)))
            fig = go.Figure()
            for lo, hi, nm in BANDS:
                fig.add_hrect(y0=lo * 100, y1=hi * 100, fillcolor=COLOR[nm], opacity=0.08, line_width=0)
            fig.add_scatter(x=h.index, y=h.score * 100, name="위험 점수(왼쪽)", mode="lines", line=dict(color="#f59e0b", width=2.4))
            fig.add_scatter(x=h.index, y=h.spx, name="S&P500(오른쪽, 로그)", mode="lines", yaxis="y2", line=dict(color="#8a99a3", width=1.6))
            fig.update_layout(yaxis=dict(range=[0, 100], title="점수"), yaxis2=dict(overlaying="y", side="right", type="log", showgrid=False))
            plot(fig, 260)
        chart()
    if kor is not None and g.get("kospi_bucket"):
        r = kor[kor.bucket == g["kospi_bucket"]]
        if len(r):
            r = r.iloc[0]
            ui.explain(
                f"<p class='note'>코스피는 지금 1년 고점보다 {kd*100:.1f}% 낮습니다. 1997년 이후 고점 대비 <b>{g['kospi_bucket']}</b> 구간이었던 때"
                f"(독립 사례 약 {int(r.episodes)}번, {int(r.days)}거래일): 이후 6개월 안에 추가로 -15% 넘게 빠진 비율 <b>{r.p_more15*100:.0f}%</b>, "
                f"12개월 뒤 수익 중앙값 <b>{r.ret12_median*100:+.0f}%</b>, 12개월 뒤 손실이었던 비율 {r.p_loss12*100:.0f}%.<br>"
                "사례가 적어 '바닥'의 근거가 되지 못합니다. 더 빠질 수 있지만 길게 보면 대체로 회복했다는 과거 기록일 뿐입니다.</p>", "ⓘ 코스피는 과거에 어땠나")
