"""AI 수익성 보고서(HTML) 생성. 숫자는 모두 data/results·data/parquet 산출물에서 읽는다(수기 입력 없음).
사용: python scripts/make_ai_report.py <출력 경로>"""
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DATA_DIR, PARQUET  # noqa: E402

R = DATA_DIR / "results"
out_path = Path(sys.argv[1])

hist = pd.read_parquet(R / "ai_history.parquet")
latest = pd.read_parquet(R / "ai_latest.parquet")
cloud_h = pd.read_parquet(R / "ai_cloud_history.parquet")
cloud_agg = pd.read_parquet(R / "ai_cloud_agg.parquet")
spreads = pd.read_parquet(R / "ai_credit_spreads.parquet")
issu = pd.read_parquet(R / "ai_credit_issuance.parquet")
tw = pd.read_parquet(PARQUET / "twse_monthly.parquet")
flags = json.loads((R / "ai_flags.json").read_text(encoding="utf-8"))
disc = pd.read_parquet(PARQUET / "ai_disclosures.parquet")
ev = pd.read_parquet(R / "ai_event_study.parquet")

b4 = hist[(hist.ticker == "BIG4")].dropna(subset=["capex_yoy"]).sort_values("period_end")
B = latest[latest.ticker == "BIG4"].iloc[0]
agg = cloud_agg.iloc[-1]
e = html.escape


def pct(v, d=0):
    return "—" if v != v else f"{v*100:+.{d}f}%"


def eok(v):  # 억 달러
    return f"{v/1e8:,.0f}억 달러"


# ---------- SVG 차트 (공통 스케일) ----------
def line_chart(series, ymin, ymax, w=640, h=250, unit="%", fmt="{:.0f}", ticks=5, xlabels=None, markers=True):
    """series: [(name, css_class, [(xi, value)...])], xi는 0..n-1 인덱스. xlabels: [(xi, text)]"""
    L, Rr, T, Bt = 44, 16, 14, 30
    n = max(x for _, _, pts in series for x, _ in pts) + 1
    sx = lambda i: L + (w - L - Rr) * (i / max(1, n - 1))  # noqa: E731
    sy = lambda v: T + (h - T - Bt) * (1 - (v - ymin) / (ymax - ymin))  # noqa: E731
    g = []
    used_labels = []
    for k in range(ticks):
        v = ymin + (ymax - ymin) * k / (ticks - 1)
        g.append(f'<line class="grid" x1="{L}" x2="{w-Rr}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                 f'<text class="tick" x="{L-6}" y="{sy(v)+4:.1f}" text-anchor="end">{fmt.format(v)}{unit}</text>')
    if ymin < 0 < ymax:
        g.append(f'<line class="zero" x1="{L}" x2="{w-Rr}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    for xi, t in (xlabels or []):
        g.append(f'<text class="tick" x="{sx(xi):.1f}" y="{h-8}" text-anchor="middle">{e(t)}</text>')
    for name, cls, pts in series:
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(v):.1f}" for j, (x, v) in enumerate(pts))
        g.append(f'<path class="ln {cls}" d="{d}"/>')
        if markers:
            for x, v in pts:
                g.append(f'<circle class="dot {cls}" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="2.6"/>')
        x, v = pts[-1]
        ly = sy(v) - 8
        while any(abs(ly - u) < 13 for u in used_labels):  # 끝값 라벨이 겹치지 않게 아래로 민다
            ly += 13
        used_labels.append(ly)
        g.append(f'<circle class="dot end {cls}" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4.4"/>'
                 f'<text class="endlabel {cls}" x="{min(sx(x)+7, w-2):.1f}" y="{ly:.1f}" text-anchor="end">{fmt.format(v)}{unit}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">{"".join(g)}</svg>'


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{e(n)}</span>' for n, c in items) + "</div>"


# 차트 A: Big4 TTM YoY
q = b4.tail(12).reset_index(drop=True)
xl = [(i, d.strftime("%y.%m")) for i, d in enumerate(q.period_end) if i % 2 == 1 or i == len(q) - 1]
chartA = line_chart([("CAPEX", "c-capex", [(i, v * 100) for i, v in enumerate(q.capex_yoy)]),
                     ("매출", "c-rev", [(i, v * 100) for i, v in enumerate(q.rev_yoy)]),
                     ("영업이익", "c-op", [(i, v * 100) for i, v in enumerate(q.opinc_yoy)])],
                    -20, 80, ticks=6, xlabels=xl)
legA = legend([("CAPEX", "c-capex"), ("매출", "c-rev"), ("영업이익", "c-op")])

# 차트 B: 클라우드 성장률
cb = cloud_h[cloud_h.quarter_end >= "2023-06-30"]
qs = sorted(cb.quarter_end.unique())
idx = {d: i for i, d in enumerate(qs)}
nm = {"AMZN": ("AWS", "c-aws"), "GOOGL": ("Google Cloud", "c-gcp"), "MSFT": ("Microsoft Intelligent Cloud", "c-ms")}
serB = [(nm[t][0], nm[t][1], [(idx[r.quarter_end], r.yoy * 100) for r in g.itertuples()]) for t, g in cb.groupby("ticker")]
chartB = line_chart(serB, 0, 100, w=460, ticks=6, xlabels=[(i, pd.Timestamp(d).strftime("%y.%m")) for d, i in idx.items() if i % 2 == 0 or i == len(qs) - 1])
legB = legend([(v[0], v[1]) for v in nm.values()])
mg = [(nm[t][0], nm[t][1], [(idx[r.quarter_end], r.margin * 100) for r in g.itertuples()]) for t, g in cb.groupby("ticker")]
chartB2 = line_chart(mg, 0, 50, w=460, ticks=6, xlabels=[(i, pd.Timestamp(d).strftime("%y.%m")) for d, i in idx.items() if i % 2 == 0 or i == len(qs) - 1])

# 차트 C: 신규 발행 10년 스프레드
sp = spreads[(spreads.bucket == "10Y") & (spreads.filed >= "2024-06-01") & (spreads.currency == "USD")].copy()
t0, t1 = pd.Timestamp("2024-06-01"), sp.filed.max()
span = (t1 - t0).days
ser = []
colors = {"GOOGL": "c-gcp", "AMZN": "c-aws", "META": "c-meta", "ORCL": "c-orcl"}
for tk, g in sp.groupby("ticker"):
    ser.append((tk, colors.get(tk, "c-ms"), [(int((r.filed - t0).days / span * 100), r.spread_bp) for r in g.itertuples()]))
chartC = line_chart(ser, 30, 110, ticks=5, unit="", fmt="{:.0f}", xlabels=[(0, "24.06"), (25, "24.12"), (50, "25.06"), (75, "25.12"), (100, t1.strftime("%y.%m"))])
legC = legend([(k, colors[k]) for k in ("GOOGL", "AMZN", "META") if k in set(sp.ticker)])

# 대만 월매출 막대
twl = tw.sort_values("month").groupby("ticker").tail(1).sort_values("yoy", ascending=False)
mx = max(130, twl.yoy.max())
bars = "".join(
    f'<div class="bar"><span class="bn">{e(r.name.split("(")[0])}</span><span class="bt"><i style="width:{max(0, r.yoy)/mx*100:.1f}%"></i></span>'
    f'<span class="bv">{r.yoy:+.0f}%</span></div>' for r in twl.itertuples())

# 신호 칩
LV = {"경고": "bad", "주의": "warn", "중립": "mid", "정상": "good", "긍정": "good", "강함": "good", "일치": "good", "보통": "mid", "둔화": "warn"}
grp = {"회수·현금": ["F1", "F2", "F5"], "수요": ["F4", "F6", "F7", "F8"], "자금조달": ["F3", "F9", "F10"]}
fl = {f["id"]: f for f in flags["flags"]}
chips = ""
for g, ids in grp.items():
    chips += f'<div class="sigcol"><h4>{g}</h4>' + "".join(
        f'<div class="chip {LV.get(fl[i]["level"], "mid")}"><b>{e(fl[i]["level"])}</b><span>{e(fl[i]["title"])}</span></div>' for i in ids if i in fl) + "</div>"

# 회사 표
rows = ""
order = ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRWV", "NVDA", "BIG4"]
lt = latest.set_index("ticker")
for t in order:
    r = lt.loc[t]
    loss = (r.op_margin == r.op_margin) and r.op_margin < 0
    rows += (f'<tr class="{"tot" if t == "BIG4" else ""}"><th>{e(r["name"])}</th><td>{r.capex_ttm/1e9:,.0f}</td><td>{pct(r.capex_yoy)}</td><td>{pct(r.rev_yoy)}</td>'
             f'<td>{"적자" if loss else pct(r.opinc_yoy)}</td><td>{r.fcf_ttm/1e9:,.0f}</td><td>{"—" if t == "NVDA" else f"{r.capex_ocf:.2f}"}</td>'
             f'<td>{"—" if r.rpo != r.rpo or t == "NVDA" else f"{r.rpo/1e9:,.0f}"}</td></tr>')

# 이벤트 스터디 요약 (NVDA·SOXX 20일, 공시 후)
evs = ev[(ev.split == "all") & (ev.h == 20) & ev.asset.isin(["SOXX", "NVDA"]) & ev.cond.isin(["capex_accelerating", "capex_decelerating"])]
ci_zero = int(((evs.ci_lo < 0) & (evs.ci_hi > 0)).sum())

# 경영진 공개(A)
d2 = disc.sort_values("filed", ascending=False)
pick = []
for kw, tk in [("run rate of \\$37 billion", "MSFT"), ("eclipsed run rates", "AMZN"), ("tokens per minute", "GOOGL")]:
    m = d2[(d2.ticker == tk) & d2.text.str.contains(kw.replace("\\$", "$"), regex=False)]
    if len(m):
        pick.append(m.iloc[0])
disc_html = "".join(f'<li><span class="who">{e(r.ticker)} · {r.quarter_end.strftime("%Y-%m")}</span> {("…" if r.text[:1].isdigit() else "") + e(r.text[:230])}</li>' for r in pick)

crwv = issu[(issu.ticker == "CRWV")].sort_values("filed")
crwv_txt = ", ".join(f"{r.coupon:.3f}% {r.currency} {int(r.maturity_year)}" for r in crwv.itertuples())
g10 = sp[sp.ticker == "GOOGL"].sort_values("filed")
a10 = sp[sp.ticker == "AMZN"].sort_values("filed")
m10 = sp[sp.ticker == "META"].sort_values("filed")

TEMPLATE = Path(__file__).with_name("ai_report_template.html").read_text(encoding="utf-8")
vals = dict(
    ASOF=latest.available.max().strftime("%Y-%m-%d"), PERIOD=B.period_end.strftime("%Y-%m"),
    CAPEX=eok(B.capex_ttm), CAPEX_YOY=pct(B.capex_yoy), REV_YOY=pct(B.rev_yoy), OP_YOY=pct(B.opinc_yoy),
    OCF=eok(B.ocf_ttm), FCF=eok(B.fcf_ttm), FCF_CHG=f"{B.fcf_chg/1e8:+,.0f}억 달러", CAPEX_OCF=f"{B.capex_ocf*100:.0f}%",
    RECOUP=f"{B.opinc_per_capex:.2f}", CLOUD_YOY=pct(agg.yoy), CLOUD_M=f"{agg.margin*100:.0f}%", CLOUD_M0=f"{agg.margin_prior*100:.0f}%",
    CLOUD_Y0=pct(cloud_agg.iloc[-5].yoy) if len(cloud_agg) >= 5 else "—",
    CHIPS=chips, CHART_A=chartA, LEG_A=legA, CHART_B=chartB, LEG_B=legB, CHART_B2=chartB2, CHART_C=chartC, LEG_C=legC,
    BARS=bars, ROWS=rows, DISC=disc_html, CRWV=e(crwv_txt),
    G10=f"{g10.spread_bp.iloc[0]:.0f}→{g10.spread_bp.iloc[-1]:.0f}bp ({g10.filed.iloc[0].strftime('%Y-%m')}→{g10.filed.iloc[-1].strftime('%Y-%m')})",
    A10=f"{a10.spread_bp.iloc[-4:].iloc[0]:.0f}→{a10.spread_bp.iloc[-1]:.0f}bp", M10=f"{m10.spread_bp.iloc[0]:.0f}→{m10.spread_bp.iloc[-1]:.0f}bp",
    TW_MONTH=tw.month.max().strftime("%Y-%m"), EV_ZERO=str(ci_zero), EV_N=str(len(evs)),
)
page = TEMPLATE
for k, v in vals.items():
    page = page.replace("{{" + k + "}}", str(v))
assert "{{" not in page, [x for x in page.split("{{")[1:3]]
out_path.write_text(page, encoding="utf-8")
print("written", out_path, len(page))
