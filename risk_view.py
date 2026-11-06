"""홈의 미국 매크로 투자 환경, 시장 스트레스 및 한국 장 마감."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from scoring.macro_outlook import METRICS
from ui import badge, load, plot

KIND = {"낮음": "good", "보통": "mid", "높음": "warn", "매우 높음": "bad"}
COLOR = {"낮음": "#22a55a", "보통": "#8bc34a", "높음": "#e08a1e", "매우 높음": "#ee5555"}
BANDS = [(0, 40, "낮음"), (40, 60, "보통"), (60, 80, "높음"), (80, 100, "매우 높음")]
OUTLOOK = {
    "caution": ("신중한 환경", "warn", "미국 주식에 우호적인 시점으로 보기 어렵습니다."),
    "mixed": ("혼재", "mid", "미국 주식에 뚜렷하게 우호적이라고 보기에는 근거가 엇갈립니다."),
    "supportive": ("우호적인 환경", "good", "현재 매크로와 시장 여건은 미국 주식에 우호적입니다."),
    "pending": ("판단 보류", "mid", "시장 또는 매크로 입력을 더 확인해야 합니다."),
}
SIGNAL = {
    "supportive": ("우호", "good"), "adverse": ("부담", "warn"), "neutral": ("중립", "mid"),
    "stale": ("갱신 지연", "warn"), "missing": ("자료 부족", "mid"),
}
WATCH = {
    "rates": "실질금리 상승세가 잦아드는지",
    "credit": "하이일드 스프레드가 다시 좁아지는지",
    "liquidity": "연준 순유동성이 감소를 멈추는지",
    "labor": "실업수당 청구 증가세가 잦아드는지",
    "inflation": "근원물가 상승률이 다시 둔화하는지",
}


def _pct(value, digits=1):
    return "—" if value is None else f"{value * 100:+.{digits}f}%"


def _stale(day, timezone, newer_day=None):
    return bool(day and (
        (datetime.now(ZoneInfo(timezone)).date() - date.fromisoformat(day)).days > 4
        or (newer_day and day < newer_day[:10])
    ))


def _macro(g, us_stale):
    macro = g.get("macro") or {}
    metrics = macro.get("metrics", {})
    outlook = "pending" if us_stale else macro.get("outlook", "pending")
    label, kind, verdict = OUTLOOK[outlook]
    buckets = {state: [METRICS[key][0] for key, metric in metrics.items() if metric["state"] == state]
               for state in SIGNAL}
    reasons = []
    if buckets["adverse"]:
        reasons.append("부담은 " + "·".join(buckets["adverse"]))
    if buckets["supportive"]:
        reasons.append("우호 요인은 " + "·".join(buckets["supportive"]))
    if not reasons and outlook == "mixed":
        reasons.append("매크로 지표에서 우호 신호가 충분히 모이지 않았습니다")
    if outlook != "pending" and (g["level"] in {"높음", "매우 높음"} or g["shock"]):
        reasons.append("미국 시장의 스트레스도 높습니다")
    detail = " · ".join(reasons)
    heading = f'<span class="t">지금 주식 투자 환경은?</span> {badge(label, kind)}'
    st.markdown(f'<div class="sum"><div>{heading} <span class="muted">미국 주식 · {g["as_of"]} 마감 기준</span></div>'
                f'<div class="head">{verdict}</div>'
                + (f'<div>{detail}</div>' if detail else "") + '</div>', unsafe_allow_html=True)

    if not metrics:
        return
    cards = []
    for key, metric in metrics.items():
        name, period, unit, _ = METRICS[key]
        state, kind = SIGNAL[metric["state"]]
        value, delta = metric["value"], metric["change"]
        if value is None:
            level = "—"
        elif key == "liquidity":
            level = f"${value / 1000:.2f}조"
        elif key == "labor":
            level = f"{value / 10000:.1f}만 건"
        else:
            level = f"{value:.2f}%"
        change = "비교 자료 없음" if delta is None else (
            f"{period} {delta:+.0f}{unit}" if unit == "bp" else f"{period} {delta:+.2f}{unit}"
        )
        if us_stale and metric["state"] not in {"missing", "stale"}:
            state, kind = "마지막 관측", "mid"
        cards.append(f'<div class="kpi"><div class="l">{name}</div><div class="v">{level}</div>'
                     f'<div class="s">{change} · {metric["as_of"] or "관측일 없음"}</div>'
                     f'<div>{badge(state, kind)}</div></div>')
    st.markdown('<div class="macro-factors kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    watch = [WATCH[key] for key, metric in metrics.items() if metric["state"] == "adverse"]
    if outlook != "pending" and (g["shock"] or g["level"] in {"높음", "매우 높음"}):
        watch.append("미국 시장의 급변과 낙폭 위험이 완화되는지")
    if watch and outlook != "pending":
        st.caption("다음 확인 포인트 · " + " · ".join(watch))


def _market_cards(g, row, us_stale, kr_stale):
    score = g["score"] * 100
    risk_badge = badge("갱신 지연", "warn") if us_stale else badge(g["level"], KIND[g["level"]])
    parts = g.get("parts", {})
    coverage = sum(v is not None for v in parts.values())
    input_note = f" · 점수 입력 {coverage}/{len(parts)}" if parts else ""
    if g.get("market_ready") is False:
        input_note += " · 달러·VIX 기준일 또는 비교 자료 확인 필요"
    prevalence = g.get("prevalence")
    frequency = ""
    if row is not None and pd.notna(row.p_bad) and prevalence is not None:
        frequency = (f'<div>같은 구간에서 이후 6개월 내 15% 이상 하락한 관측일 <b>{row.p_bad:.0%}</b> '
                     f'· 전체 {prevalence:.0%}</div>')
    risk = (f'<div class="ev"><div class="title">미국 중기 낙폭 위험 {risk_badge}</div>'
            f'<div class="big">{score:.0f}<span class="muted"> / 100</span></div>'
            f'<div class="bigl">{g["as_of"]} 마감{input_note}</div>'
            f'<div>S&amp;P 500 {g["spx"]:,.2f} · 1년 최고 종가 대비 {_pct(g["spx_dd"])}</div>{frequency}</div>')

    trigger = g.get("shock_trigger")
    alert = badge("갱신 지연", "warn") if us_stale else (
        badge("급변", "warn") if g["shock"] else
        badge("자료 확인 필요", "mid") if g.get("market_ready") is False else badge("미발동", "mid")
    )
    trigger_info = ""
    if trigger:
        hits = []
        if trigger["spx_5d"] is not None:
            hits.append(f'S&amp;P 500 {_pct(trigger["spx_5d"])}')
        if trigger["vix_5d"] is not None:
            hits.append(f'VIX {_pct(trigger["vix_5d"], 0)}')
        trigger_info = f'<div><b>{trigger["as_of"]} 최근 기준 충족</b> · {" · ".join(hits)}</div>'
    elif g["shock"]:
        trigger_info = '<div>최근 5거래일 안에 경보 기준 충족</div>'
    vix = f'{g["vix"]:.1f}' if g.get("vix") is not None else "—"
    vix_date = g.get("vix_as_of", g["as_of"]) or "관측일 없음"
    vix_change = _pct(g.get("vix5"), 0)
    shock = (f'<div class="ev"><div class="title">미국 단기 급변 {alert}</div>'
             f'<div class="big"><span class="muted">VIX</span> {vix}</div>'
             f'<div class="bigl">{vix_date} VIX 관측</div>'
             f'<div>최근 5거래일 · S&amp;P 500 {_pct(g.get("ret5"))} · VIX {vix_change}</div>{trigger_info}</div>')

    close, previous = g.get("kospi_level"), g.get("kospi_prev_close")
    kret = close / previous - 1 if close is not None and previous else None
    klevel = f"{close:,.2f}" if close is not None else "—"
    kday = g.get("kospi_as_of")
    kbadge = badge("갱신 지연", "warn") if kr_stale else ""
    korea = (f'<div class="ev"><div class="title">한국 장 마감 {kbadge}</div>'
             f'<div class="big">{klevel}</div>'
             f'<div class="bigl">코스피 · {kday or "기준일 확인 중"}</div>'
             f'<div>당일 <b>{_pct(kret, 2)}</b> · 1년 최고 종가 대비 {_pct(g.get("kospi_dd"))}</div></div>')
    ui.trio(risk, shock, korea)


def render(meta=None):
    g = load("results/risk_gauge.json")
    if g is None:
        st.info("매크로 투자 환경과 시장 위험 데이터를 준비하고 있습니다.")
        return
    meta = meta or {}
    us_stale = _stale(g["as_of"], "America/New_York", meta.get("us_close"))
    kr_stale = _stale(g.get("kospi_as_of"), "Asia/Seoul", meta.get("kis_last"))
    lv = load("results/risk_gauge_levels.parquet")
    levels = lv.loc[lv.level == g["level"]] if lv is not None else None
    row = levels.iloc[0] if levels is not None and len(levels) else None
    _macro(g, us_stale)
    _market_cards(g, row, us_stale, kr_stale)

    with st.expander("판단 기준과 위험 점수 추이"):
        st.markdown(
            "**매크로 판정**은 주식 보유에 영향을 주는 환경을 요약합니다. "
            "부담 신호가 3개 이상이거나 2개 이상이면서 미국 시장 스트레스가 높으면 신중, "
            "우호 신호가 3개 이상이고 부담과 높은 시장 스트레스가 모두 없으면 우호, 그 외에는 혼재로 봅니다. "
            "금리·신용·물가·실업수당의 상승과 순유동성의 감소를 부담 방향으로 봅니다. "
            "기준 변화폭은 표시 순서대로 ±25bp, ±20bp, ±1%, ±5%, ±0.2%p입니다. "
            "연준 순유동성은 연준 자산에서 재무부 일반계정과 역레포를 뺀 값입니다. "
            "실업수당과 근원 CPI는 최초 발표값을 사용합니다. "
            "이 기준의 수익률 예측력은 검증 전이며, 주식의 가격 매력도는 별도 평가가 필요합니다."
        )
        st.markdown(
            "**미국 중기 위험**은 5개 시장 지표의 과거 대비 위치를 평균한 점수입니다. "
            "이후 126거래일 중 오늘 종가 대비 15% 이상 하락한 과거 빈도를 함께 보여줍니다. "
            "관측 기간이 겹치고 독립 급락 사례가 적습니다. 2007년 이후 구분력(AUC 약 0.56)도 제한적입니다. "
            "**미국 급변**은 S&P 500의 5거래일 수익률 −5% 이하 또는 VIX의 5거래일 변화 +60% 이상을 "
            "최근 5거래일 중 한 번이라도 충족하면 표시합니다."
        )
        if lv is not None:
            st.caption(" · ".join(f"{r.level} {r.p_bad:.0%} ({int(r.days):,}거래일)"
                                  for r in lv.dropna(subset=["p_bad", "days"]).itertuples()))
        ss = g.get("shock_stats") or {}
        if ss.get("events") and ss.get("p_bad") is not None:
            st.caption(f'과거 미국 급변 {ss["events"]}건 중 이후 6개월 내 15% 이상 하락한 비율 {ss["p_bad"]:.0%} · '
                       f'6개월 뒤 수익률 중앙값 {_pct(ss.get("ret6_median"), 0)}')
        _history()

    kor = load("results/risk_gauge_kospi.parquet")
    if kor is not None and g.get("kospi_bucket"):
        rows = kor[kor.bucket == g["kospi_bucket"]]
        if len(rows):
            r = rows.iloc[0]
            ui.explain(
                f'<p class="note">1997년 이후 코스피의 1년 고점 대비 낙폭이 <b>{r.bucket}</b>였던 관측일 '
                f'{int(r.days):,}일(독립 사례 약 {int(r.episodes)}건) 중 6개월 내 추가 15% 이상 하락한 비율은 '
                f'<b>{r.p_more15:.0%}</b>, 12개월 뒤 수익률 중앙값은 <b>{r.ret12_median:+.0%}</b>였습니다. '
                '같은 급락기에 속한 날짜가 여럿 포함된 과거 기록입니다.</p>', "ⓘ 코스피는 과거에 어땠나")


@st.fragment
def _history():
    hist = load("results/risk_gauge_history.parquet")
    if hist is None or hist.empty:
        return
    rng = st.segmented_control("기간", ["3년", "10년", "2005년 이후"], default="3년", key="rg_rng")
    h = hist.tail({"3년": 156, "10년": 520}.get(rng or "3년", len(hist)))
    fig = go.Figure()
    for lo, hi, name in BANDS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=COLOR[name], opacity=0.08, line_width=0)
    fig.add_scatter(x=h.index, y=h.score * 100, name="위험 점수(왼쪽)", mode="lines", line=dict(color="#f59e0b", width=2.4))
    fig.add_scatter(x=h.index, y=h.spx, name="S&P 500(오른쪽, 로그)", mode="lines", yaxis="y2", line=dict(color="#8a99a3", width=1.6))
    fig.update_layout(yaxis=dict(range=[0, 100], title="점수"), yaxis2=dict(overlaying="y", side="right", type="log", showgrid=False))
    plot(fig, 260)
