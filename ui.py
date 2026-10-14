"""공통 UI: 스타일, 카드·배지, 데이터 로더(수정시각 기준 캐시), 차트 스타일.
앱은 data/views(작은 요약)와 data/results(작은 분석 결과)만 읽는다."""
import json

import plotly.graph_objects as go
import streamlit as st

from src.data import DATA_DIR

CSS = """
<style>
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1320px}
h1,h2,h3{letter-spacing:-0.01em}
.lede{font-size:1.12rem;line-height:1.7;margin:.2rem 0 1rem}
.muted{color:#8a99a3;font-size:.86rem}
.b{display:inline-block;padding:2px 10px;border-radius:10px;font-size:.76rem;font-weight:700;margin-right:6px}
.b.good{background:#15803d22;color:#22a55a;border:1px solid #22a55a55}
.b.warn{background:#d9770622;color:#e08a1e;border:1px solid #e08a1e66}
.b.bad{background:#dc262622;color:#ee5555;border:1px solid #ee555566}
.b.mid{background:#64748b22;color:#8a99a3;border:1px solid #64748b55}
.b.info{background:#2563eb22;color:#5b8def;border:1px solid #5b8def66}
.card{border:1px solid #64748b44;border-radius:12px;padding:18px 20px;height:100%}
.card h3{margin:0 0 4px;font-size:1.08rem}
.card .head{font-size:1.02rem;line-height:1.65;margin:8px 0 14px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.stat{border-left:3px solid #64748b55;padding:2px 0 2px 10px;min-width:0}
.stat .l{font-size:.76rem;color:#8a99a3}.stat .v{font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}.stat .s{font-size:.78rem;color:#8a99a3}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;align-items:stretch}
.step{border:1px solid #64748b44;border-radius:12px;padding:14px 16px;position:relative;min-width:0}
.step .n{font-size:.74rem;color:#8a99a3;font-weight:700;letter-spacing:.06em}
.step .t{font-size:1.02rem;font-weight:700;margin:2px 0 8px}
.step .kv{font-size:.86rem;line-height:1.55}
.step.good{border-top:3px solid #22a55a}.step.warn{border-top:3px solid #e08a1e}.step.bad{border-top:3px solid #ee5555}.step.mid{border-top:3px solid #64748b}
.fresh{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.8rem;color:#8a99a3;margin:.3rem 0 1rem}
.note{font-size:.84rem;color:#8a99a3;line-height:1.6}
div[data-testid="stMetricValue"]{font-size:1.6rem}
@media (max-width:1000px){.stats{grid-template-columns:1fr}.chain{grid-template-columns:1fr 1fr}}
</style>
"""


def css():
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text, kind="mid"):
    return f'<span class="b {kind}">{text}</span>'


LEVEL = {"경고": "bad", "주의": "warn", "긍정": "good", "강함": "good", "정상": "good", "일치": "good", "중립": "mid", "보통": "mid", "둔화": "warn"}


def level_kind(level: str) -> str:
    if "괴리" in level:
        return "warn"
    return LEVEL.get(level, "mid")


@st.cache_data(show_spinner=False, max_entries=64)
def _read(path: str, mtime: float):
    import pandas as pd
    if path.endswith(".json"):
        return json.loads(open(path, encoding="utf-8").read())
    return pd.read_parquet(path)


def load(rel: str):
    """rel 예: 'views/state.json', 'results/ai_latest.parquet'. 파일이 바뀌면(mtime) 자동으로 다시 읽는다."""
    p = DATA_DIR / rel
    if not p.exists():
        return None
    return _read(str(p), p.stat().st_mtime)


def load_parquet_dir(rel: str):
    from src.data import PARQUET
    p = PARQUET / rel
    if not p.exists():
        return None
    return _read(str(p), p.stat().st_mtime)


def fig_style(fig: go.Figure, h: int = 300, legend=True) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=8, b=0), showlegend=legend,
                      legend=dict(orientation="h", y=1.14, x=0), hovermode="x unified",
                      font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(100,116,139,0.18)", zeroline=False)
    return fig


def plot(fig: go.Figure, h: int = 300, legend=True, key=None):
    st.plotly_chart(fig_style(fig, h, legend), width="stretch", key=key,
                    config={"displayModeBar": False, "responsive": True})


def card(title: str, badge_html: str, headline: str, stats: list):
    s = "".join(f'<div class="stat"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{sub}</div></div>' for l, v, sub in stats)
    st.markdown(f'<div class="card"><h3>{title}</h3>{badge_html}<div class="head">{headline}</div><div class="stats">{s}</div></div>',
                unsafe_allow_html=True)


def freshness(meta: dict):
    items = [("미국 마감", meta["us_close"]), ("한국 수급", meta["kis_last"]), ("SEC 최신 공시", meta["sec_filed"]),
             ("재무 기준 분기", meta["sec_period"]), ("대만 월매출", meta["twse_month"][:7])]
    st.markdown('<div class="fresh">' + "".join(f"<span>{k} <b>{v}</b></span>" for k, v in items) + "</div>", unsafe_allow_html=True)


def pct(v, d=0, sign=True):
    if v is None or v != v:
        return "—"
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"
