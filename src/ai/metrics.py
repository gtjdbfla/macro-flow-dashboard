"""AI 인프라 투자 수익성 지표 (SEC 공시 기반, point-in-time).

모든 지표는 **회사 전체 재무**에서 계산한다. AI 전용 수치가 아니다(MSFT·GOOGL·AMZN·META의 CAPEX에는 일반 데이터센터가 섞이고,
매출·이익에는 광고·유통 등 비AI 사업이 섞인다). 그래서 신뢰 등급은 전부 'C'(AI 귀속은 간접 추정)이고, RPO(수주잔고)만 'B'
(클라우드 계약 수요의 직접 지표)다. 점수 합성은 하지 않는다(가중치를 검증할 표본이 없다) — 원지표·추세·자기이력 대비 위치를 그대로 보여준다.

availability: 분기 값은 구성 항목(CAPEX·OCF·매출)이 모두 공시된 날(filed 최댓값) 이후에만 쓴다."""
import numpy as np
import pandas as pd

from src.data import PARQUET

BIG4 = ["MSFT", "GOOGL", "AMZN", "META"]
INFRA = ["ORCL", "CRWV"]
FLOWS = ["capex", "ocf", "revenue", "op_income", "net_income", "da", "buyback"]
NAMES = {"MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "ORCL": "Oracle",
         "CRWV": "CoreWeave", "NVDA": "NVIDIA", "BIG4": "Big4 합계"}


def load(asof=None) -> pd.DataFrame:
    d = pd.read_parquet(PARQUET / "sec_quarterly.parquet")
    if asof is not None:
        d = d[d["filed"] <= pd.Timestamp(asof)]
    return d


def wide(d: pd.DataFrame, ticker: str):
    """(분기말 인덱스) 값 표와 availability(공시일) 시리즈."""
    t = d[d.ticker == ticker]
    v = t.pivot_table(index="end", columns="metric", values="value", aggfunc="last").sort_index()
    f = t[t.metric.isin(["capex", "ocf", "revenue"])].pivot_table(index="end", columns="metric", values="filed",
                                                                    aggfunc="max").sort_index()
    need = [c for c in ("capex", "ocf", "revenue") if c in f]
    avail = f[need].max(axis=1) if len(need) == 3 else pd.Series(pd.NaT, index=v.index)
    return v, avail.reindex(v.index)


def derive(v: pd.DataFrame) -> pd.DataFrame:
    """분기 표 -> TTM·성장률·효율 지표. 분기가 연속(약 3개월 간격)이어야 TTM을 만든다."""
    g = v.index.to_series().diff().dt.days
    contiguous4 = g.rolling(3).apply(lambda x: ((x > 80) & (x < 105)).all(), raw=True).fillna(0).astype(bool)
    o = pd.DataFrame(index=v.index)

    def ttm(c):
        s = v[c].rolling(4).sum() if c in v else pd.Series(np.nan, index=v.index)
        return s.where(contiguous4)

    for c in FLOWS:
        o[f"{c}_ttm"] = ttm(c)
    o["fcf_ttm"] = o["ocf_ttm"] - o["capex_ttm"]
    yoy = lambda s: (s / s.shift(4) - 1).where(s.shift(4) > 0)  # noqa: E731
    o["capex_yoy"] = yoy(o["capex_ttm"])
    o["rev_yoy"] = yoy(o["revenue_ttm"])
    o["rev_q_yoy"] = yoy(v["revenue"]) if "revenue" in v else np.nan
    o["opinc_yoy"] = yoy(o["op_income_ttm"])
    o["ocf_yoy"] = yoy(o["ocf_ttm"])
    o["da_yoy"] = yoy(o["da_ttm"])
    o["fcf_chg"] = o["fcf_ttm"] - o["fcf_ttm"].shift(4)  # 부호가 바뀔 수 있어 증가율 대신 금액 변화
    o["capex_ocf"] = o["capex_ttm"] / o["ocf_ttm"].where(o["ocf_ttm"] > 0)
    o["capex_rev"] = o["capex_ttm"] / o["revenue_ttm"].where(o["revenue_ttm"] > 0)
    o["fcf_margin"] = o["fcf_ttm"] / o["revenue_ttm"].where(o["revenue_ttm"] > 0)
    o["op_margin"] = o["op_income_ttm"] / o["revenue_ttm"].where(o["revenue_ttm"] > 0)
    o["da_gap"] = o["da_yoy"] - o["rev_yoy"]  # 감가상각(투자 비용) 증가가 매출 증가를 앞서는 정도
    # 1년 전 CAPEX 1달러당 늘어난 매출·영업이익(투자 회수 속도의 거친 대리지표)
    lag_capex = o["capex_ttm"].shift(4).where(o["capex_ttm"].shift(4) > 0)
    o["rev_per_capex"] = (o["revenue_ttm"] - o["revenue_ttm"].shift(4)) / lag_capex
    o["opinc_per_capex"] = (o["op_income_ttm"] - o["op_income_ttm"].shift(4)) / lag_capex
    if "rpo" in v:
        o["rpo"] = v["rpo"]
        o["rpo_yoy"] = yoy(v["rpo"])
        o["rpo_cov"] = v["rpo"] / o["revenue_ttm"].where(o["revenue_ttm"] > 0)
    if "lt_debt" in v:
        o["lt_debt"] = v["lt_debt"]
        o["debt_yoy"] = yoy(v["lt_debt"])
    o["payout"] = o["buyback_ttm"] / o["ocf_ttm"].where(o["ocf_ttm"] > 0)
    return o


def all_companies(asof=None) -> dict:
    """{ticker: (derived 표, availability)} — BIG4 합계 포함."""
    d = load(asof)
    out = {}
    for t in BIG4 + INFRA + ["NVDA"]:
        if t not in set(d.ticker):
            continue
        v, a = wide(d, t)
        out[t] = (derive(v), a)
    # Big4 합계: 4사 모두 값이 있는 분기말만
    vs = {t: wide(d, t) for t in BIG4 if t in set(d.ticker)}
    if len(vs) == 4:
        cols = FLOWS + ["rpo", "lt_debt"]
        idx = sorted(set.intersection(*[set(v.index[a.notna()]) for v, a in vs.values()]))
        agg = pd.DataFrame(index=pd.DatetimeIndex(idx))
        for c in cols:
            parts = [v[c].reindex(agg.index) for v, _ in vs.values() if c in v]
            # RPO는 AMZN·META 값이 없어 합산이 왜곡되므로 4사 모두 있을 때만
            if c in ("rpo", "lt_debt") and len(parts) < 4:
                continue
            agg[c] = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
        out["BIG4"] = (derive(agg), pd.concat([a.reindex(agg.index) for _, a in vs.values()], axis=1).max(axis=1))
        # RPO 보유 3사(MSFT·GOOGL·ORCL) 별도 합계는 대시보드에서 개별 표시
    return out


def latest_table(co: dict) -> pd.DataFrame:
    """회사별 최신 분기 지표 표."""
    rows = []
    for t, (o, a) in co.items():
        x = o.dropna(subset=["capex_ttm", "ocf_ttm"]).tail(1)
        if x.empty:
            continue
        r = x.iloc[0]
        rows.append({"ticker": t, "name": NAMES.get(t, t), "period_end": x.index[0], "available": a.loc[x.index[0]],
                     **{k: r.get(k, np.nan) for k in ["capex_ttm", "capex_yoy", "rev_yoy", "opinc_yoy", "ocf_ttm",
                                                      "fcf_ttm", "fcf_chg", "capex_ocf", "capex_rev", "fcf_margin",
                                                      "op_margin", "da_gap", "rev_per_capex", "opinc_per_capex",
                                                      "rpo", "rpo_yoy", "rpo_cov", "debt_yoy", "payout"]}})
    return pd.DataFrame(rows)


def flags(co: dict, prices: dict) -> list:
    """사전 등록 규칙(docs/AI_PROFITABILITY.md)에 따른 상태 플래그. 임계값은 수익률을 보고 고르지 않았다."""
    out = []
    b = co.get("BIG4")
    if b is not None:
        o = b[0].dropna(subset=["capex_yoy"]).iloc[-1]
        n = b[0].dropna(subset=["capex_yoy"]).index[-1]
        f1 = bool(o["capex_yoy"] > 0.30 and o["fcf_chg"] < 0)
        out.append({"id": "F1", "title": "투자 확대 vs 현금창출", "on": f1, "level": "경고" if f1 else "정상",
                    "text": f"Big4 CAPEX TTM {o['capex_yoy']*100:+.0f}% YoY, FCF TTM 변화 {o['fcf_chg']/1e8:+,.0f}억 달러 — "
                            + ("투자는 30%+ 늘었는데 잉여현금흐름은 줄었다." if f1 else "FCF가 줄지 않았거나 CAPEX 증가가 30% 미만.")})
        h = b[0].dropna(subset=["opinc_per_capex"])
        pct = float((h["opinc_per_capex"] < h["opinc_per_capex"].iloc[-1]).mean())
        f2 = pct < 1 / 3
        out.append({"id": "F2", "title": "투자 회수 속도(영업이익)", "on": f2, "level": "경고" if f2 else "정상",
                    "text": f"1년 전 CAPEX 1달러당 영업이익 증가 {h['opinc_per_capex'].iloc[-1]:.2f}달러 — 자기 이력(2014~) 대비 하위 {pct*100:.0f}%"
                            + (" (하위 1/3: 회수 둔화)" if f2 else "")})
    lev = []
    for t in INFRA:
        if t in co:
            x = co[t][0].dropna(subset=["capex_ocf"]).tail(1)
            if len(x) and (x["capex_ocf"].iloc[0] > 1.0 or (x.get("debt_yoy", pd.Series([0])).iloc[0] > 0.3 and x["capex_ocf"].iloc[0] > 0.8)):
                lev.append(f"{NAMES[t]} CAPEX/OCF {x['capex_ocf'].iloc[0]:.2f}")
    out.append({"id": "F3", "title": "차입 의존 투자(Oracle·CoreWeave)", "on": bool(lev), "level": "주의" if lev else "정상",
                "text": ("; ".join(lev) + " — CAPEX가 영업현금흐름을 넘거나 부채가 급증하며 근접.") if lev else "두 회사 모두 CAPEX/OCF 1 미만이거나 부채 급증 조건 미충족."})
    rp = []
    for t in ("MSFT", "GOOGL", "ORCL"):
        if t in co and "rpo_yoy" in co[t][0]:
            x = co[t][0].dropna(subset=["rpo_yoy", "rev_yoy"]).tail(1)
            if len(x):
                rp.append((NAMES[t], float(x["rpo_yoy"].iloc[0]), float(x["rev_yoy"].iloc[0])))
    ahead = [r for r in rp if r[1] > r[2]]
    out.append({"id": "F4", "title": "수요 선행(수주잔고) vs 매출", "on": len(ahead) == len(rp) and len(rp) > 0,
                "level": "긍정" if rp and len(ahead) == len(rp) else "중립",
                "text": ", ".join(f"{n} RPO {a*100:+.0f}% vs 매출 {r*100:+.0f}%" for n, a, r in rp)
                        + " — 수주잔고 증가가 매출 증가를 앞서면 계약 수요가 선행."})
    # AI 수요 수취자(NVIDIA): 투자의 반대편에서 매출이 실제로 늘고 있는가
    if "NVDA" in co:
        x = co["NVDA"][0].dropna(subset=["rev_yoy"]).tail(1)
        if len(x):
            ry = float(x["rev_yoy"].iloc[0])
            oy = float(x["ocf_yoy"].iloc[0]) if "ocf_yoy" in x else float("nan")
            strong = ry > 0.30
            out.append({"id": "F6", "title": "AI 수요 측(NVIDIA 매출·현금흐름)", "on": strong, "level": "강함" if strong else "둔화",
                        "text": "NVIDIA 매출 TTM %+.0f%% YoY, 영업현금흐름 TTM %+.0f%% YoY — " % (ry * 100, oy * 100)
                                + ("하이퍼스케일러 투자가 공급사 매출로 실제 전환 중." if strong else "공급사 매출 성장이 30% 아래로 둔화.")})
    # 시장-펀더멘털 괴리: 반도체 주가 vs Big4 투자·현금흐름
    if b is not None and prices:
        dd = {k: float(v.iloc[-1] / v.tail(252).max() - 1) for k, v in prices.items() if len(v) > 252}
        worst = min(dd.values()) if dd else 0
        o = b[0].dropna(subset=["capex_yoy"]).iloc[-1]
        fundamentals_ok = bool(o["capex_yoy"] > 0.10 and not out[0]["on"])
        if worst <= -0.15 and fundamentals_ok:
            lvl, txt = "괴리(주가↓·펀더멘털 유지)", "반도체 주가는 52주 고점 대비 −15% 이하인데 Big4 투자는 늘고 FCF는 줄지 않았다."
        elif max(dd.values(), default=0) >= -0.05 and (out[0]["on"] or (len(out) > 1 and out[1]["on"])):
            lvl, txt = "괴리(주가 낙관·수익성 신호 약화)", "반도체 주가는 고점 부근인데 투자 대비 현금창출·회수 지표가 약해졌다."
        else:
            lvl, txt = "일치", "주가와 투자·현금흐름 신호 사이에 뚜렷한 괴리는 없다."
        out.append({"id": "F5", "title": "시장 vs 펀더멘털", "on": lvl != "일치", "level": lvl,
                    "text": txt + " 52주 고점 대비: " + ", ".join(f"{k} {v*100:+.0f}%" for k, v in dd.items())})
    return out
