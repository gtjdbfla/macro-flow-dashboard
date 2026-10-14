"""갱신 현황: 자동 갱신이 언제·무엇을·어떻게 돌고 있는지, 데이터 소스별 신선도."""
import json

import pandas as pd
import streamlit as st

import ui
from ui import load

src = load("views/sources.parquet")
sched = load("views/schedule.json")
stt = load("status.json")

st.title("갱신 현황")
st.markdown('<p class="lede">사람의 개입 없이 서버가 평일 하루 두 번 데이터를 받아 분석과 화면을 새로 만듭니다. '
            '아래 표에서 <b>언제 돌고, 다음 실행은 언제이며, 데이터 소스별로 최신 상태인지</b> 확인합니다. 시간은 모두 한국 시간(KST)입니다.</p>', unsafe_allow_html=True)

n_ok = int((src.status == "정상").sum())
bad = src[src.status != "정상"]
c = st.columns(3)
c[0].metric("정상 데이터 소스", f"{n_ok} / {len(src)}")
c[1].metric("확인 필요", f"{len(bad)}")
c[1].caption(", ".join(bad.source) if len(bad) else "지연·오류 없음")
last_full = None if stt is None else (pd.Timestamp(stt["finished"]) + pd.Timedelta(hours=9)).strftime("%m-%d %H:%M")
c[2].metric("마지막 전체 갱신(KST)", last_full or "아직 없음")
c[2].caption("" if stt is None else ("정상 완료" if stt["ok"] else "문제 있음"))
if stt and not stt["ok"]:
    st.warning("마지막 갱신에 문제가 있었습니다: " + "; ".join(stt["warnings"])[:400])

KIND = {"정상": "#15803d33", "지연": "#d9770644", "일부 오류": "#d9770644", "오류": "#dc262644", "데이터 없음": "#64748b44", "성공": "#15803d33", "실패": "#dc262644"}


def color(v):
    return f"background-color: {KIND[v]}" if v in KIND else ""


def styled(df, cols):
    s = df.style
    return (s.map if hasattr(s, "map") else s.applymap)(color, subset=cols)


NAME = {"kr": "한국 수급", "us": "미국 마감 후", "weekly": "주간 재계산", "filings": "실적 시즌 점검"}
st.header("1. 실행 일정")
sd = pd.DataFrame([{"구분": NAME[s["mode"]], "하는 일": s["what"], "일정": s["when"], "다음 실행": s["next_kst"], "마지막 실행": s["last_kst"],
                    "결과": "—" if s["last_ok"] is None else ("성공" if s["last_ok"] else "실패"),
                    "소요(초)": s["last_s"], "비고": s["last_warn"]} for s in sched])
st.dataframe(styled(sd, ["결과"]), hide_index=True, width="stretch")
st.markdown('<p class="note">백테스트·검증 통계는 하루 새 데이터로는 거의 바뀌지 않아 주 1회만 다시 계산합니다. 실적 시즌에는 Big4 실적 발표일이 늦지 않게 반영되도록 3시간마다 새 공시만 확인하고, '
            '새 공시가 있을 때만 분석을 갱신합니다. 대만 월매출은 매월 10~12일에만 수집합니다.</p>', unsafe_allow_html=True)

st.header("2. 데이터 소스별 현황")
t = src.copy()
t["last_kst"] = [("—" if not x else (pd.Timestamp(x) + pd.Timedelta(hours=9)).strftime("%m-%d %H:%M")) for x in t.last_run_utc]
tb = pd.DataFrame({"데이터": t.content, "갱신 주기": t.cadence, "마지막 수집": t.last_kst, "데이터 최신일": t.data_end,
                   "경과일": t.age_days, "허용 경과일": t.tolerance, "상태": t.status, "비고": t.note})
st.dataframe(styled(tb, ["상태"]), hide_index=True, width="stretch", height=410)
st.markdown('<p class="note"><b>경과일</b>은 오늘과 데이터 최신일의 차이, <b>허용 경과일</b>은 정상으로 보는 한도입니다(주말·휴장·공표 지연 포함). '
            '한도를 넘으면 지연, 시리즈 오류가 많으면 오류로 표시합니다. 재무·클라우드·채권 공시는 분기·비정기라 경과일이 길어도 정상입니다.</p>', unsafe_allow_html=True)

st.header("3. 최근 실행 이력")
hp = ui.DATA_DIR / "status_history.jsonl"
if hp.exists():
    rows = [json.loads(x) for x in hp.read_text(encoding="utf-8").splitlines()[-20:]]
    h = pd.DataFrame(rows).iloc[::-1]
    out = pd.DataFrame({"시작(KST)": [(pd.Timestamp(x) + pd.Timedelta(hours=9)).strftime("%m-%d %H:%M") for x in h.started],
                        "구분": h["mode"].map(NAME), "소요(초)": h.duration_s, "결과": h.ok.map(lambda v: "성공" if v else "실패"),
                        "주의": h.warnings.map(lambda w: "; ".join(w))})
    st.dataframe(styled(out, ["결과"]), hide_index=True, width="stretch", height=300)
if stt:
    with st.expander("마지막 전체 갱신의 단계별 결과"):
        st.dataframe(pd.DataFrame(stt["steps"]).rename(columns={"name": "단계", "ok": "성공", "seconds": "초", "tail": "오류 내용"}), hide_index=True, width="stretch")
st.markdown('<p class="note">실패하면 이전 결과를 그대로 두고 이 페이지와 홈 상단에 경고를 띄웁니다. 운영 방법: docs/OPERATIONS.md.</p>', unsafe_allow_html=True)
