"""공통 UI: 스타일, 요약 카드·KPI 타일·근거 카드, 데이터 로더(수정시각 기준 캐시), 차트 스타일.
앱은 data/views(작은 요약)와 data/results(작은 분석 결과)만 읽는다.

레이아웃 원칙(PC 우선): 본문 16px · 한 줄 70자 안팎 · 문단 대신 '큰 숫자 + 짧은 설명' 카드 · 같은 줄의 카드는 높이를 맞춘다 ·
표는 잘리지 않을 폭에 둔다 · 경고·한계는 하나의 callout으로 모은다."""
import json
import re

import plotly.graph_objects as go
import streamlit as st

from src.data import DATA_DIR

CSS = """
<style>
html{font-size:16px}
.block-container{max-width:1440px;padding:5rem 2.6rem 4rem}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{font-size:1rem;line-height:1.72}
h1{font-size:2.1rem!important;margin-bottom:.2rem!important}
h2{font-size:1.5rem!important;margin:1.6rem 0 .6rem!important}
h3{font-size:1.2rem!important;margin:1.2rem 0 .5rem!important}
.lede{font-size:1.08rem;line-height:1.7;margin:.1rem 0 .8rem;max-width:62rem}
.muted,.note{color:#8a99a3;font-size:.88rem;line-height:1.6}
.b{display:inline-block;padding:2px 10px;border-radius:10px;font-size:.78rem;font-weight:700;margin-right:6px;white-space:nowrap}
.b.good{background:#15803d22;color:#22a55a;border:1px solid #22a55a55}
.b.warn{background:#d9770622;color:#e08a1e;border:1px solid #e08a1e66}
.b.bad{background:#dc262622;color:#ee5555;border:1px solid #ee555566}
.b.mid{background:#64748b22;color:#8a99a3;border:1px solid #64748b55}
.b.info{background:#2563eb22;color:#5b8def;border:1px solid #5b8def66}
/* 요약 카드 한 쌍: grid로 높이를 맞춘다 */
.duo{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:.4rem 0 1rem}
.sum{border:1px solid #64748b44;border-radius:14px;padding:22px 26px;display:flex;flex-direction:column;gap:14px;min-width:0}
.sum .t{font-size:1.05rem;font-weight:700;letter-spacing:-.01em}
.sum .head{font-size:1.14rem;line-height:1.62;font-weight:500}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:auto}
.kpi{background:#64748b14;border-radius:10px;padding:12px 14px;min-width:0}
.kpi .l{font-size:.8rem;color:#8a99a3}
.kpi .v{font-size:1.5rem;font-weight:800;line-height:1.3;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kpi .s{font-size:.82rem;color:#8a99a3}
/* 근거 카드(등급 배지 + 한 줄 결론 + 큰 숫자 + 구간별 줄 + 칩) */
.trio{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:.4rem 0 1rem}
.ev{border:1px solid #64748b44;border-radius:14px;padding:20px 22px;display:flex;flex-direction:column;gap:10px;min-width:0}
.ev .title{font-size:1.06rem;font-weight:700;line-height:1.45}
.ev .big{font-size:2.1rem;font-weight:800;line-height:1.15;font-variant-numeric:tabular-nums}
.ev .bigl{font-size:.84rem;color:#8a99a3;margin-top:-6px}
.ev .rows{display:grid;grid-template-columns:1fr auto auto;gap:3px 12px;font-size:.9rem;font-variant-numeric:tabular-nums;border-top:1px solid #64748b33;padding-top:10px}
.ev .rows span:nth-child(3n+2),.ev .rows span:nth-child(3n){text-align:right}
.ev .chipl{font-size:.8rem;color:#8a99a3;margin-top:4px}
.chip{display:inline-block;padding:3px 10px;border-radius:8px;font-size:.88rem;margin:2px 6px 2px 0}
.chip.up{background:#22a55a22;color:#3fc57a}.chip.dn{background:#ee555522;color:#f07575}.chip.nu{background:#64748b22;color:#aab6bd}
.callout{border-left:4px solid #e08a1e;background:#e08a1e12;padding:12px 16px;border-radius:0 8px 8px 0;margin:.6rem 0 1rem;font-size:.95rem;line-height:1.65}
.callout.info{border-left-color:#5b8def;background:#5b8def12}
.fresh{display:flex;flex-wrap:wrap;gap:4px 22px;font-size:.84rem;color:#8a99a3;margin:.2rem 0 .6rem}
.fresh b{color:#c0cbd2}
.mini{width:100%;border-collapse:collapse;font-size:.92rem;font-variant-numeric:tabular-nums}
.mini th{text-align:left;color:#8a99a3;font-weight:600;font-size:.8rem;padding:4px 8px;border-bottom:1px solid #64748b44}
.mini td{padding:5px 8px;border-bottom:1px solid #64748b22}
.mini td.r,.mini th.r{text-align:right}
.hold td{padding:8px;vertical-align:top;line-height:2}.hold .chip{margin:2px 4px 2px 0}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;align-items:stretch}
.step{border:1px solid #64748b44;border-radius:12px;padding:16px 18px;min-width:0}
.step .n{font-size:.76rem;color:#8a99a3;font-weight:700;letter-spacing:.06em}
.step .t{font-size:1.04rem;font-weight:700;margin:2px 0 8px}
.step .kv{font-size:.9rem;line-height:1.6}
.step.good{border-top:3px solid #22a55a}.step.warn{border-top:3px solid #e08a1e}.step.bad{border-top:3px solid #ee5555}.step.mid{border-top:3px solid #64748b}
/* 과거 → 현재 → 다음: 모든 페이지가 같은 순서로 읽히도록 단계 표지를 쓴다 */
.stage{display:flex;align-items:center;gap:12px;font-size:1.3rem;font-weight:700;margin:2.2rem 0 .7rem;padding-top:1.1rem;border-top:1px solid #64748b33;letter-spacing:-.01em}
.stage .s{font-size:.82rem}
.tag{display:inline-block;padding:3px 12px;border-radius:8px;font-size:.82rem;font-weight:800;letter-spacing:.02em;white-space:nowrap}
.tag.past{background:#64748b33;color:#b4c0c8}.tag.now{background:#2563eb2e;color:#7fa6f5}.tag.next{background:#e08a1e2e;color:#f0a53c}
.purpose{border:1px solid #64748b44;border-radius:14px;padding:20px 24px;margin:.2rem 0 1rem;display:flex;flex-direction:column;gap:12px}
.purpose .q{font-size:1.25rem;font-weight:800;letter-spacing:-.01em}
.purpose .a{font-size:1.06rem;line-height:1.62}
.flow3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.flow3 div.c{background:#64748b12;border-radius:10px;padding:10px 14px;font-size:.92rem;line-height:1.55;color:#c0cbd2}
.flow3 div.c .tag{margin-bottom:5px}
.legend{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:center;font-size:.86rem;color:#8a99a3;margin:.1rem 0 .6rem}
div[data-testid="stMetricValue"]{font-size:1.6rem}
@media (max-width:1100px){.duo,.trio,.chain,.flow3{grid-template-columns:1fr}.kpis{grid-template-columns:1fr}}
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
                      legend=dict(orientation="h", y=1.14, x=0), hovermode="x unified", font=dict(size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(100,116,139,0.18)", zeroline=False)
    return fig


def plot(fig: go.Figure, h: int = 300, legend=True, key=None):
    st.plotly_chart(fig_style(fig, h, legend), width="stretch", key=key, config={"displayModeBar": False, "responsive": True})


def kpis(items):
    """[(라벨, 값, 보조)] -> KPI 타일 HTML."""
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for l, v, s in items) + "</div>"


def summary_card(title, badges, headline, items):
    return (f'<div class="sum"><div><span class="t">{title}</span></div><div>{badges}</div>'
            f'<div class="head">{headline}</div>{kpis(items)}</div>')


def duo(a: str, b: str):
    st.markdown(f'<div class="duo">{a}{b}</div>', unsafe_allow_html=True)


def ev_card(badge_html, title, big, big_label, rows, chips_html=""):
    """rows: [(라벨, 값, 표시)] — 구간별 결과 줄. chips_html: 후보 칩."""
    r = "".join(f"<span>{a}</span><span>{b}</span><span>{c}</span>" for a, b, c in rows)
    return (f'<div class="ev"><div>{badge_html}</div><div class="title">{title}</div><div class="big">{big}</div>'
            f'<div class="bigl">{big_label}</div><div class="rows">{r}</div>{chips_html}</div>')


def trio(a, b, c):
    st.markdown(f'<div class="trio">{a}{b}{c}</div>', unsafe_allow_html=True)


def chips(label, names, kind):
    return f'<div class="chipl">{label}</div><div>' + "".join(f'<span class="chip {kind}">{n}</span>' for n in names) + "</div>"


def callout(text, kind=""):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


TAGS = {"과거": "past", "현재": "now", "다음": "next"}


def stage(tag: str, title: str, sub: str = ""):
    """단계 표지: [과거|현재|다음] + 제목(+ 부연). 페이지마다 같은 순서(과거 → 현재 → 다음)로 쓴다."""
    s = f'<span class="s muted">{sub}</span>' if sub else ""
    st.markdown(f'<div class="stage"><span class="tag {TAGS[tag]}">{tag}</span><span>{title}</span>{s}</div>', unsafe_allow_html=True)


def purpose(question: str, answer: str, steps, badges: str = ""):
    """페이지 맨 위 목적 배너: 질문 → 한 줄 답 → 이 페이지가 보여주는 과거/현재/다음. steps=[(태그, 설명)]."""
    cells = "".join(f'<div class="c"><span class="tag {TAGS[t]}">{t}</span><br>{d}</div>' for t, d in steps)
    st.markdown(f'<div class="purpose"><div class="q">{question}</div><div>{badges}</div><div class="a">{answer}</div><div class="flow3">{cells}</div></div>',
                unsafe_allow_html=True)


def legend():
    st.markdown('<div class="legend"><span>근거 등급</span>' + badge("검증됨", "good") + "세 구간(2005–18·2019–22·2023~)에서 같은 방향" + badge("단서", "warn") +
                "방향은 보이나 약함" + badge("참고", "mid") + "근거 부족·현황만</div>", unsafe_allow_html=True)


_STRIP = re.compile(r"(?<![A-Za-z])(INC|CORP|CORPORATION|CO|PLC|LTD|HOLDINGS?|CLASS [A-C]|CL [A-C]|THE|COMPANY|SHARES)(?![A-Za-z])\.?", re.I)


def short_name(n: str) -> str:
    """ETF 보유 종목명의 법인 접미사·주식 종류를 떼어 짧게."""
    t = _STRIP.sub(" ", n.replace("/", " ").replace("+", "&")).strip(" &")
    return " ".join(w.capitalize() if len(w) > 3 else w for w in t.split())


def holdings_table(hd, sector_order):
    """섹터별 대표 종목 표. hd: sector_holdings 뷰, sector_order: 표시 순서(섹터명 리스트)."""
    rows = []
    for sec_name in sector_order:
        g = hd[hd.sector == sec_name].sort_values("rank")
        if g.empty:
            continue
        cells = "".join(f'<span class="chip nu"><b>{r.ticker}</b> {short_name(r.name)} {r.weight:.1f}%</span>' for r in g.head(5).itertuples())
        rows.append(f"<tr><td><b>{sec_name}</b><br><span class='muted'>{g.etf.iloc[0]} · 보유 {int(g.top_n.iloc[0])}종목</span></td>"
                    f"<td>{cells}</td><td class='r'>{g.head(5).weight.sum():.0f}%</td></tr>")
    return ("<table class='mini hold'><tr><th>섹터</th><th>대표 종목 (ETF 내 비중)</th><th class='r'>상위 5 합</th></tr>" + "".join(rows) + "</table>")


def freshness(meta: dict):
    items = [("미국 마감", meta["us_close"]), ("한국 수급", meta["kis_last"]), ("SEC 최신 공시", meta["sec_filed"]),
             ("재무 기준 분기", meta["sec_period"]), ("대만 월매출", meta["twse_month"][:7])]
    st.markdown('<div class="fresh">' + "".join(f"<span>{k} <b>{v}</b></span>" for k, v in items) + "</div>", unsafe_allow_html=True)


def pct(v, d=0, sign=True):
    if v is None or v != v:
        return "—"
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"
