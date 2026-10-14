"""근거 등급 재평가: 더 큰 표본·더 강한 검정으로 각 주장을 같은 규칙으로 다시 판정한다.

판정 규칙(결과를 보기 전에 고정하지 못한 부분은 docs/EVIDENCE_GRADES.md에 그대로 적는다):
  검증됨  : 세 구간 모두 부호가 같고, 그중 2개 이상에서 |t|>=2, 전체(합산) |t|>=3
  단서    : 전체 |t|>=2 이고 세 구간 중 2개 이상이 같은 부호
  참고    : 그 밖의 모든 경우
t는 Newey-West(겹치는 수익률 길이만큼 lag). 횡단면 신호는 매 시점 스피어만 IC를 낸 뒤 시간 평균한다.
주의: 이 평가는 이미 본 과거 데이터에 대한 것이다. '전향 확인'(scoring/ledger.py)이 쌓이기 전에는 '과거 기준 검증됨'이다."""
import glob
import os
import warnings

import numpy as np
import pandas as pd

from backtest.forward_returns import Fwd
from src.data import DATA_DIR, PARQUET
from src.features.common import Ctx
from src.features.sector_features import SECTORS

warnings.filterwarnings("ignore")

US_SPLITS = {"train": ("2005-01-01", "2018-12-31"), "val": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2100-01-01")}
IND_SPLITS = {"train": ("1927-01-01", "1962-12-31"), "val": ("1963-01-01", "1999-12-31"), "test": ("2000-01-01", "2100-01-01")}
FAC_SPLITS = {"train": ("1963-07-01", "1989-12-31"), "val": ("1990-01-01", "2009-12-31"), "test": ("2010-01-01", "2100-01-01")}
AI_SPLITS = {"train": ("2010-01-01", "2018-12-31"), "val": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2100-01-01")}


def nw_t(x, lag):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan, np.nan, n
    m = x.mean()
    u = x - m
    s = (u @ u) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return float(m), float(m / np.sqrt(s / n)), n


def spearman_ic(sig: pd.DataFrame, fwd: pd.DataFrame, minn: int) -> pd.Series:
    sig, fwd = sig.align(fwd, join="inner")
    ok = sig.notna() & fwd.notna()
    n = ok.sum(axis=1)
    rs, rf = sig.rank(axis=1).where(ok), fwd.rank(axis=1).where(ok)
    rs, rf = rs.sub(rs.mean(axis=1), axis=0), rf.sub(rf.mean(axis=1), axis=0)
    ic = (rs * rf).sum(axis=1) / np.sqrt((rs ** 2).sum(axis=1) * (rf ** 2).sum(axis=1))
    return ic.where(n >= minn)


def by_split(s: pd.Series, splits: dict, h: int, lag: int | None = None) -> dict:
    """구간별 평균·t. 구간 끝에서 h개(미래 윈도우가 구간을 넘는 행)는 버린다(purge)."""
    out = {}
    for k, (a, b) in splits.items():
        x = s[(s.index >= a) & (s.index <= b)].dropna()
        x = x.iloc[: max(0, len(x) - h)]
        m, t, n = nw_t(x.values, lag if lag is not None else h)
        out[k] = {"mean": m, "t": t, "n": n}
    allx = pd.concat([s[(s.index >= a) & (s.index <= b)].dropna().iloc[: max(0, len(s[(s.index >= a) & (s.index <= b)].dropna()) - h)] for a, b in splits.values()])
    m, t, n = nw_t(allx.values, lag if lag is not None else h)
    out["all"] = {"mean": m, "t": t, "n": n}
    return out


def grade(st: dict, expect_sign: int | None = None) -> tuple[str, str]:
    """st: by_split 결과. expect_sign: 사전에 기대한 부호(+1/-1)가 있으면 그 부호 기준, 없으면 합산 부호 기준."""
    ks = ["train", "val", "test"]
    ma = st["all"]["mean"]
    if not np.isfinite(ma):
        return "참고", "표본 부족"
    sg = expect_sign if expect_sign else int(np.sign(ma))
    same = [k for k in ks if np.isfinite(st[k]["mean"]) and np.sign(st[k]["mean"]) == sg]
    sig = [k for k in same if abs(st[k]["t"]) >= 2]
    ta = abs(st["all"]["t"])
    if len(same) == 3 and len(sig) >= 2 and ta >= 3 and np.sign(ma) == sg:
        return "검증됨", f"세 구간 같은 부호, |t|≥2가 {len(sig)}개 구간, 전체 t {st['all']['t']:+.1f}"
    if ta >= 2 and len(same) >= 2 and np.sign(ma) == sg:
        diff = [k for k in ks if k not in same]
        return "단서", f"전체 t {st['all']['t']:+.1f}, 같은 부호 {len(same)}/3 구간" + (f" (어긋난 구간: {', '.join(diff)})" if diff else "")
    return "참고", f"전체 t {st['all']['t']:+.1f}, 같은 부호 {len(same)}/3 구간"


PRIMARY = {("us_flow_contrarian", 20), ("kr_foreign_persistence", 20), ("industry_momentum", 1), ("factor_momentum", 1),
           ("us_sector_momentum", 20), ("ai_capex_leadlag", 4)}


LABS = {"industry_momentum": ("1927–62", "1963–99", "2000~"), "factor_momentum": ("1963–89", "1990–2009", "2010~"),
        "ai_capex_leadlag": ("2010–18", "2019–22", "2023~")}


def _row(claim, test, h, st, g, why, unit, primary=None):
    """primary: 등급 판정에 쓰는 대표 검정인가. 한 주장을 여러 호라이즌으로 쳐서 합격한 것만 고르는 일을 막기 위해, 대표는 사전에 하나로 고정한다."""
    prim = ((claim, h) in PRIMARY) if primary is None else primary
    t_recent = st["test"]["t"]
    recent = ""
    if g == "검증됨" and np.isfinite(t_recent) and abs(t_recent) < 2:
        recent = f"최근 구간 t {t_recent:+.1f}: 약화"
    lab = LABS.get(claim, ("2005–18", "2019–22", "2023~"))
    return {"claim": claim, "test": test, "h": h, "unit": unit, "grade": g, "why": why, "primary": prim, "recent": recent,
            "lab_train": lab[0], "lab_val": lab[1], "lab_test": lab[2],
            **{f"{k}_{f}": st[k][f] for k in ("train", "val", "test", "all") for f in ("mean", "t", "n")}}


# ── 미국 섹터 ETF 자금 ───────────────────────────────────────────────────────
def us_flow(ctx, fw, S, labs):
    rows = []
    idx = ctx.cal
    for H in (20, 5, 60):
        Y = pd.DataFrame({lab: fw.rel(sym, "SPY", H)[0] for sym, lab in labs.items()}, index=idx)
        X = pd.DataFrame({lab: S[f"sector__{lab}__flow_20d"] for lab in labs.values()})
        Z = (X - X.expanding(250).mean().shift(1)) / X.expanding(250).std().shift(1)  # 섹터 고유 수준 제거, 룩어헤드 없음
        ic = spearman_ic(Z, Y[X.columns], 8)
        st = by_split(ic, US_SPLITS, H)
        g, why = grade(st, expect_sign=-1)
        rows.append(_row("us_flow_contrarian", f"11섹터 20일 순유입(자기대비 z)→{H}일 후 SPY 대비 (IC)", H, st, g, why, "IC"))
    return rows


def us_flow_conditions(ctx, fw, S, labs):
    """2019–22에서만 부호가 반대인 이유 탐색(사후 분해라 탐색적). 조건 3개를 미리 정하고 모두 보고한다."""
    idx = ctx.cal
    H = 20
    Y = pd.DataFrame({lab: fw.rel(sym, "SPY", H)[0] for sym, lab in labs.items()}, index=idx)
    X = pd.DataFrame({lab: S[f"sector__{lab}__flow_20d"] for lab in labs.values()})
    Z = (X - X.expanding(250).mean().shift(1)) / X.expanding(250).std().shift(1)
    ic = spearman_ic(Z, Y[X.columns], 8)
    risk = pd.read_parquet(PARQUET / "features_risk.parquet").reindex(idx)
    spy = ctx.close("SPY")
    conds = {"VIX 순위 높음(≥0.5)": (risk["risk__VIX__level_rank"] >= 0.5), "VIX 순위 낮음(<0.5)": (risk["risk__VIX__level_rank"] < 0.5),
             "SPY 200일선 위": (spy > spy.rolling(200).mean()), "SPY 200일선 아래": (spy <= spy.rolling(200).mean()),
             "섹터 간 자금 분산 큼(상위절반)": (X.std(axis=1) > X.std(axis=1).expanding(250).median().shift(1)),
             "섹터 간 자금 분산 작음": (X.std(axis=1) <= X.std(axis=1).expanding(250).median().shift(1))}
    rows = []
    for name, m in conds.items():
        s = ic.where(m.reindex(ic.index).fillna(False))
        st = by_split(s, US_SPLITS, H)
        rows.append({"condition": name, **{f"{k}_{f}": st[k][f] for k in ("train", "val", "test", "all") for f in ("mean", "t", "n")}})
    return rows


# ── 한국 종목 단위 외국인 수급 ────────────────────────────────────────────────
def kr_stock_panel():
    px, fr, tv = {}, {}, {}
    for p in glob.glob(str(PARQUET / "kis_stock_*.parquet")):
        t = os.path.basename(p)[10:16]
        d = pd.read_parquet(p)
        d = d[d.stck_clpr > 0]
        px[t], fr[t], tv[t] = d.stck_clpr, d.frgn_ntby_tr_pbmn, d.acml_tr_pbmn / 1e6
    px, fr, tv = (pd.DataFrame(x).sort_index() for x in (px, fr, tv))
    rows = []
    for H in (20, 5, 60):
        ent = px.shift(-1)
        y = ent.shift(-H) / ent - 1
        Y = y.sub(y.mean(axis=1), axis=0)
        n = 60
        X = (fr.rolling(n, min_periods=48).sum() / tv.rolling(n, min_periods=48).sum()).where(px.notna())
        ic = spearman_ic(X, Y, 8)
        st = by_split(ic, US_SPLITS, H)
        g, why = grade(st, expect_sign=1)
        rows.append(_row("kr_foreign_persistence", f"한국 28종목 외국인 60일 순매수 비중→{H}일 후 종목 간 상대수익 (IC)", H, st, g, why, "IC"))
    return rows


# ── 100년 장기: 산업·팩터 모멘텀 ───────────────────────────────────────────────
def long_history():
    rows, now = [], {}
    ind = pd.read_parquet(PARQUET / "french_ind.parquet")
    ff = pd.read_parquet(PARQUET / "french_ff.parquet")
    mom = pd.read_parquet(PARQUET / "french_mom.parquet")
    lg = np.log1p(ind)
    for nm, w, skip in (("12−1개월", 12, 1), ("6−1개월", 6, 1)):
        X = lg.rolling(w).sum().shift(skip)
        for h in (1, 3):
            Y = np.expm1(lg.rolling(h).sum().shift(-h))
            Y = Y.sub(Y.mean(axis=1), axis=0)
            ic = spearman_ic(X, Y, 30)
            st = by_split(ic, IND_SPLITS, h)
            g, why = grade(st, expect_sign=1)
            rows.append(_row("industry_momentum", f"49산업 과거 {nm} 수익 순위→{h}개월 후 산업 간 상대수익 (IC)", h, st, g, why, "IC",
                             primary=(nm == "12−1개월" and h == 1)))
    F = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(mom.rename(columns={mom.columns[0]: "Mom"})).dropna(how="all")
    p12 = F.rolling(12).sum()
    ts = (np.sign(p12) * F.shift(-1))              # 과거 12개월 승자 팩터는 롱, 패자 팩터는 숏
    for nm, cols in (("모멘텀 제외 5팩터 평균", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]), ("6팩터 평균", list(F.columns))):
        s = ts[cols].mean(axis=1).where(p12[cols].notna().all(axis=1))
        st = by_split(s, FAC_SPLITS, 1)
        g, why = grade(st, expect_sign=1)
        rows.append(_row("factor_momentum", f"팩터 시계열 모멘텀({nm}): 과거 12개월 부호→다음 달 수익", 1, st, g, why, "월 수익(소수)",
                         primary=nm.startswith("모멘텀 제외")))
    # 지금 신호(가장 최근 월말 기준)
    last = ind.index[-1]
    r12 = lg.rolling(12).sum().shift(1).loc[last].dropna().sort_values(ascending=False)
    now["asof"] = str(last.date())
    now["industry_top"] = [(k, float(np.expm1(v))) for k, v in r12.head(5).items()]
    now["industry_bottom"] = [(k, float(np.expm1(v))) for k, v in r12.tail(5).items()]
    now["factor_12m"] = {k: float(v) for k, v in F.rolling(12).sum().iloc[-1].items()}
    return rows, now


# ── 미국 섹터 ETF 가격 모멘텀(ETF 시대 재확인) ─────────────────────────────────
def us_sector_momentum(ctx, fw, S, labs):
    rows = []
    idx = ctx.cal
    for H in (20,):
        Y = pd.DataFrame({lab: fw.rel(sym, "SPY", H)[0] for sym, lab in labs.items()}, index=idx)
        X = pd.DataFrame({lab: S[f"sector__{lab}__mom_12_1"] for lab in labs.values()})
        ic = spearman_ic(X, Y[X.columns], 8)
        st = by_split(ic, US_SPLITS, H)
        g, why = grade(st, expect_sign=1)
        rows.append(_row("us_sector_momentum", f"11섹터 12−1개월 모멘텀→{H}일 후 SPY 대비 (IC)", H, st, g, why, "IC"))
    return rows


# ── AI: 투자(CAPEX) → 이후 매출·이익 ──────────────────────────────────────────
def ai_leadlag():
    h = pd.read_parquet(DATA_DIR / "results" / "ai_history.parquet")
    h = h[h.ticker.isin(["AMZN", "GOOGL", "MSFT", "META", "ORCL"])].copy()
    h = h.sort_values(["ticker", "period_end"])
    rows = []
    specs = [("capex_yoy", "rev_yoy", "CAPEX 증가율(t)→매출 증가율(t+4분기)", 4), ("capex_yoy", "opinc_yoy", "CAPEX 증가율(t)→영업이익 증가율(t+4분기)", 4),
             ("capex_rev", "op_margin", "CAPEX/매출(t)→영업이익률 변화(t→t+4분기)", 4)]
    for xc, yc, name, k in specs:
        parts = []
        for tk, g in h.groupby("ticker"):
            g = g.set_index("period_end")
            g.index = pd.DatetimeIndex(g.index) + pd.offsets.QuarterEnd(0)   # Oracle(5월 결산)을 달력 분기에 맞춘다
            g = g[~g.index.duplicated(keep="last")]
            x = g[xc]
            y = g[yc].shift(-k) - (g[yc] if yc == "op_margin" else 0)
            parts.append(pd.DataFrame({"x": x, "y": y, "tk": tk}))
        d = pd.concat(parts).dropna()
        d["x"] = d.groupby("tk").x.transform(lambda s: (s - s.mean()) / s.std())
        d["y"] = d.groupby("tk").y.transform(lambda s: (s - s.mean()) / s.std())
        # 모든 회사가 함께 오르내리는 시간 추세로 생기는 가짜 상관을 막기 위해 분기 평균도 뺀다(회사 간 상대 비교)
        d["x"] = d.x - d.groupby(level=0).x.transform("mean")
        d["y"] = d.y - d.groupby(level=0).y.transform("mean")
        d = d[d.groupby(level=0).x.transform("size") >= 4]
        s = (d.x * d.y).groupby(level=0).mean()          # 분기별 5사 평균 상관 기여
        st = by_split(s, AI_SPLITS, k, lag=k)
        g_, why = grade(st, expect_sign=1)
        rows.append(_row("ai_capex_leadlag", name, k, st, g_, why + " · 5개사 소표본", "상관(표준화)", primary=(xc, yc) == ("capex_yoy", "rev_yoy")))
    return rows


# ── 한국 전달 모델 강화 ───────────────────────────────────────────────────────
def korea_model_v2(ctx):
    """후보 입력 4세트를 사전에 정하고: train(~2018)으로 적합 → val(2019–22) 성과로 하나를 고르고 → train+val 재적합 후 test(2023~)를 본다."""
    K = pd.read_parquet(PARQUET / "features_korea.parquet").reindex(ctx.cal)
    M = pd.read_parquet(PARQUET / "features_macro.parquet").reindex(ctx.cal)
    R = pd.read_parquet(PARQUET / "features_risk.parquet").reindex(ctx.cal)
    ret = lambda s: ctx.close(s).pct_change(fill_method=None)  # noqa: E731
    X = pd.DataFrame({"QQQ": ret("QQQ"), "SOXX": ret("SOXX"), "DXY": ret("DX-Y.NYB"), "d_Real10Y": M["macro__Real10Y__chg_1d"],
                      "KRW": ret("KRW=X"), "HYG": ret("HYG"), "dVIX": R["risk__VIX__chg_1d"],
                      "lag_foreign_z": K["kr__KOSPI_foreign__net_1d_z"]})
    y = K["kr__KOSPI_foreign__net_1d_z"].shift(-1)
    sets = {"A 기존(4개)": ["QQQ", "SOXX", "DXY", "d_Real10Y"], "B +원/달러": ["QQQ", "SOXX", "DXY", "d_Real10Y", "KRW"],
            "C +직전 외국인 z": ["QQQ", "SOXX", "DXY", "d_Real10Y", "lag_foreign_z"],
            "D +원/달러+HYG+VIX+직전z": ["QQQ", "SOXX", "DXY", "d_Real10Y", "KRW", "HYG", "dVIX", "lag_foreign_z"]}
    idx = ctx.cal

    def fit(cols, a, b):
        m = (idx >= a) & (idx <= b)
        d = pd.concat([X[cols], y.rename("y")], axis=1)[m].dropna()
        mu, sd = d[cols].mean(), d[cols].std()
        A = np.c_[np.ones(len(d)), ((d[cols] - mu) / sd).values]
        beta, *_ = np.linalg.lstsq(A, d["y"].values, rcond=None)
        return mu, sd, beta

    def ev(cols, a, b, mu, sd, beta):
        m = (idx >= a) & (idx <= b)
        Z = ((X[cols] - mu) / sd)[m]
        e = pd.concat([Z, y[m].rename("y")], axis=1).dropna()
        p = np.c_[np.ones(len(e)), e[cols].values] @ beta
        yy = e["y"].values
        return float(np.corrcoef(p, yy)[0, 1]), float(1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()), len(e)

    rows = []
    for nm, cols in sets.items():
        mu, sd, b1 = fit(cols, "2005-01-01", "2018-12-31")
        cv, rv, nv = ev(cols, "2019-01-01", "2022-12-31", mu, sd, b1)
        mu2, sd2, b2 = fit(cols, "2005-01-01", "2022-12-31")
        ct, rt, nt = ev(cols, "2023-01-01", "2100-01-01", mu2, sd2, b2)
        rows.append({"set": nm, "val_corr": cv, "val_r2": rv, "val_n": nv, "test_corr": ct, "test_r2": rt, "test_n": nt})
    return rows


def run():
    ctx = Ctx()
    fw = Fwd(ctx)
    S = pd.read_parquet(PARQUET / "features_sector.parquet").reindex(ctx.cal)
    labs = {sym: lab for sym, lab in SECTORS.items() if fw.close(sym) is not None}
    rows = []
    rows += us_flow(ctx, fw, S, labs)
    rows += us_sector_momentum(ctx, fw, S, labs)
    rows += kr_stock_panel()
    lh, now = long_history()
    rows += lh
    rows += ai_leadlag()
    tests = pd.DataFrame(rows)
    cond = pd.DataFrame(us_flow_conditions(ctx, fw, S, labs))
    k2 = pd.DataFrame(korea_model_v2(ctx))
    return tests, cond, k2, now
