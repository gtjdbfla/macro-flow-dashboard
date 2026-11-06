"""클라우드 부문(Azure·AWS·Google Cloud) 분석 — 발표문 파싱 데이터(src/data/cloud.py) 기반.
YoY·마진은 각 발표문 안의 '당기 vs 전년 동기'로 계산(부문 재편성 영향 최소화, MSFT는 2024-08 재편)."""
import numpy as np
import pandas as pd

from src.data import PARQUET

NAMES = {"MSFT": "Microsoft Intelligent Cloud", "GOOGL": "Google Cloud", "AMZN": "AWS"}


def history() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET / "cloud_segments.parquet").sort_values(["ticker", "quarter_end"]).copy()
    d["yoy"] = d.cloud_rev / d.cloud_rev_prior - 1
    d["margin"] = d.cloud_opinc / d.cloud_rev
    d["margin_prior"] = d.cloud_opinc_prior / d.cloud_rev_prior
    d["margin_chg_pp"] = (d.margin - d.margin_prior) * 100
    d["opinc_yoy"] = d.cloud_opinc / d.cloud_opinc_prior - 1
    d["yoy_chg_pp"] = d.groupby("ticker")["yoy"].diff() * 100  # 직전 발표 대비 성장률 변화(가속/감속)
    return d


def aggregate(h: pd.DataFrame) -> pd.DataFrame:
    """3사 합계(세 회사 값이 모두 있는 분기)."""
    metrics = ["cloud_rev", "cloud_rev_prior", "cloud_opinc", "cloud_opinc_prior"]
    p = h.pivot_table(index="quarter_end", columns="ticker", values=metrics)
    columns = pd.MultiIndex.from_product([metrics, NAMES])
    p = p.reindex(columns=columns).dropna()
    out = pd.DataFrame(index=p.index)
    out["rev"] = p["cloud_rev"].sum(axis=1)
    out["rev_prior"] = p["cloud_rev_prior"].sum(axis=1)
    out["yoy"] = out.rev / out.rev_prior - 1
    out["margin"] = p["cloud_opinc"].sum(axis=1) / out.rev
    out["margin_prior"] = p["cloud_opinc_prior"].sum(axis=1) / out.rev_prior
    return out


def latest(h: pd.DataFrame, co: dict) -> pd.DataFrame:
    rows = []
    for t, g in h.groupby("ticker"):
        r = g.iloc[-1]
        capex_lag = np.nan
        if t in co:
            o = co[t][0]
            q = r.quarter_end - pd.DateOffset(years=1)
            same = o.index[(abs((o.index - q).days) < 20)]
            if len(same):
                capex_lag = o.loc[same[0], "capex_ttm"]
        rows.append({"ticker": t, "name": NAMES[t], "quarter_end": r.quarter_end, "rev_q": r.cloud_rev, "yoy": r.yoy,
                     "yoy_chg_pp": r.yoy_chg_pp, "opinc_q": r.cloud_opinc, "margin": r.margin, "margin_chg_pp": r.margin_chg_pp,
                     "azure_yoy": r.get("azure_yoy", np.nan),
                     # 연환산 클라우드 매출 증가 ÷ 1년 전 같은 회사 전체 CAPEX TTM(거친 비교: 분모는 비클라우드 CAPEX 포함)
                     "ann_rev_gain_per_capex": (r.cloud_rev - r.cloud_rev_prior) * 4 * 1e6 / capex_lag if capex_lag == capex_lag and capex_lag > 0 else np.nan})
    return pd.DataFrame(rows)


def flags(h: pd.DataFrame, tw: pd.DataFrame | None) -> list:
    """F7 클라우드 성장·마진, F8 대만 AI 공급망 월매출. 임계값은 docs/AI_PROFITABILITY.md에 사전 고정."""
    out = []
    last = h.sort_values("quarter_end").groupby("ticker").tail(1)
    current = last[last.quarter_end == last.quarter_end.max()]
    complete = current.dropna(subset=["yoy", "yoy_chg_pp", "margin", "margin_chg_pp"])
    acc = int((complete.yoy_chg_pp > 0).sum())
    mar = int((complete.margin_chg_pp > 0).sum())
    if len(complete) < 2:
        lvl = "보류"
    elif acc >= 2 and mar >= 2:
        lvl = "긍정"
    elif acc >= 2:
        lvl = "중립"
    else:
        lvl = "둔화"
    txt = [f"{NAMES[r.ticker]} {r.quarter_end:%Y-%m} 성장 {r.yoy:+.0%} · 마진 {r.margin:.0%}"
           for r in complete.itertuples()]
    azure = last.loc[last.ticker == "MSFT"].dropna(subset=["azure_yoy"]) if "azure_yoy" in last else last.iloc[:0]
    if len(azure):
        r = azure.iloc[0]
        txt.append(f"Azure {r.quarter_end:%Y-%m} 매출 {r.azure_yoy:+.0%}(부문 마진 비교와 별도)")
    summary = f"비교 가능한 {len(complete)}곳 중 {acc}곳 성장 가속, {mar}곳 마진 개선"
    out.append({"id": "F7", "title": "클라우드 매출 가속·수익성", "on": lvl == "긍정", "level": lvl,
                "text": " · ".join([summary, *txt])})
    if tw is not None and len(tw):
        core = ["2330", "2317", "2382", "6669", "3231", "2356", "2376", "2345"]
        m = tw[tw.ticker.isin(core)].sort_values("month").groupby("ticker").tail(1)
        share = float((m.yoy > 20).mean())
        lvl = "강함" if share >= 0.75 else ("보통" if share >= 0.5 else "둔화")
        top = ", ".join("%s %+.0f%%" % (r.name.split("(")[0], r.yoy) for r in m.sort_values("yoy", ascending=False).head(4).itertuples())
        out.append({"id": "F8", "title": "대만 AI 공급망 월매출(선행)", "on": lvl == "강함", "level": lvl,
                    "text": "%s월 기준 핵심 8사 중 %.0f%%가 YoY +20%% 초과 — %s" % (m.month.max().strftime("%Y-%m"), share * 100, top)})
    return out
