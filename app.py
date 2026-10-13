"""Macro & Capital Flow Dashboard — Phase 1: 데이터 현황 화면(분석 화면은 백테스트 이후)."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import DATA_DIR, PARQUET

st.set_page_config(page_title="Macro & Capital Flow", layout="wide")
st.title("Macro & Capital Flow")
st.caption("Phase 1 — 데이터 파이프라인. 분석·점수 화면은 백테스트 검증 후에 붙는다.")

st_path = PARQUET / "features_state.parquet"
if st_path.exists():
    S = pd.read_parquet(st_path).dropna(subset=["state_key"])
    cur = S.iloc[-1]
    st.subheader(f"현재 매크로 상태 ({S.index[-1].date()})")
    c = st.columns(5)
    for col, (lab, key) in zip(c, [("Regime", "regime"), ("Growth", "growth"), ("Inflation", "inflation"),
                                   ("Policy", "policy"), ("Liquidity", "liquidity")]):
        col.metric(lab, cur[key])
    st.caption(f"Dollar: {cur['dollar']} · 같은 state_key를 가진 과거 일수: "
               f"{int((S['state_key'] == cur['state_key']).sum())} / {len(S)}  (규칙 고정, 튜닝 없음 — Phase 3에서 검증)")

cov_path = DATA_DIR / "coverage.json"
if not cov_path.exists():
    st.warning("아직 수집된 데이터가 없습니다 (scripts/collect.py).")
    st.stop()
cov = json.loads(cov_path.read_text(encoding="utf-8"))

st.subheader("소스별 커버리지")
rows = []
for src, v in cov.items():
    for k, x in v.get("series", {}).items():
        rows.append({"source": src, "series": k, "start": x.get("start"), "end": x.get("end"),
                     "n": x.get("n"), "error": x.get("error")})
    if v.get("error"):
        rows.append({"source": src, "series": "(전체)", "error": v["error"]})
    st.caption(f"{src}: 갱신 {v.get('updated')}")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

kp = PARQUET / "kis_investor_kospi.parquet"
if kp.exists():
    st.subheader("KOSPI 투자자별 순매수 (KIS, 백만원) — 20일 누적")
    d = pd.read_parquet(kp)
    fig = go.Figure()
    for c, n in [("frgn_ntby_tr_pbmn", "외국인"), ("orgn_ntby_tr_pbmn", "기관"), ("prsn_ntby_tr_pbmn", "개인")]:
        fig.add_scatter(x=d.index, y=d[c].rolling(20).sum(), name=n)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
