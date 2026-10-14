"""Macro & Capital Flow — 두 질문에 답하는 대시보드.
  ① 월가의 자금은 지금 어디에 있고, 과거에 어떻게 움직였나   ② AI 기업의 수익성은 어떻게 변하고 있나
페이지: 홈 · 월가 자금 · AI 수익성 · 근거·방법. 앱은 data/views(작은 요약)만 읽는다(scripts/build_views.py)."""
import streamlit as st

import ui

st.set_page_config(page_title="Macro & Capital Flow", layout="wide", initial_sidebar_state="collapsed")
ui.css()
pages = [
    st.Page("views/home.py", title="홈", default=True),
    st.Page("views/flows.py", title="월가 자금"),
    st.Page("views/ai.py", title="AI 수익성"),
    st.Page("views/evidence.py", title="근거·방법"),
]
st.navigation(pages, position="top").run()
