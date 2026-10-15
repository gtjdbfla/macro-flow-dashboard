"""AI 투자 기업의 재무 안전성(신용) 지표 — CDS 시세 대신 무료로 계산할 수 있는 대용 지표 묶음.

CDS·유통시장 채권 수익률은 유료라 수집하지 않는다. 대신
  1) 대차대조표·이자·현금흐름에서 나오는 신용 지표(순부채, 이자보상배율, 12개월 유동성 런웨이, 계약잔고(RPO)/부채)
  2) 주가·변동성·부채로 계산하는 Merton 모형의 시장 내재 신용 스프레드(5년) — 수준은 CDS가 아니라 '자기 이력 대비 위치'로 본다
  3) ICE BofA 신용등급대별 OAS(BBB·BB·B·CCC, FRED 최근 3년) — AI 차입자가 속한 시장의 금융 환경
  4) 증분 ROIC: 2022년 대비 늘어난 CAPEX 대비 영업이익 증가(수익성 판단)
을 계산한다. 전부 SEC 공시(공시일 이후에만 사용)와 공개 가격으로 만든다."""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

from src.data import DATA_DIR, PARQUET
from src.data.yahoo import _fname

NAMES = {"MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "ORCL": "Oracle", "CRWV": "CoreWeave", "NVDA": "NVIDIA"}
BORROWERS = ["ORCL", "CRWV"]          # 차입 의존도가 높은 AI 인프라 기업(안전성 판단의 중심)
TAX = 0.17                             # 증분 ROIC 계산용 가정 세율
HURDLE = 0.10                          # 가정 자본비용


def _q() -> pd.DataFrame:
    return pd.read_parquet(PARQUET / "sec_quarterly.parquet")


def _latest(q, tk, metric, asof=None):
    d = q[(q.ticker == tk) & (q.metric == metric)]
    if asof is not None:
        d = d[d.filed <= asof]
    if d.empty:
        return None
    r = d.sort_values(["end", "filed"]).iloc[-1]
    return pd.Timestamp(r.end), float(r.value)


def balance_sheet(q, tk, asof=None) -> dict:
    """공시일(asof 이전) 기준 최신 대차대조표 항목. 항목별 기준일을 함께 반환한다."""
    g = lambda m: _latest(q, tk, m, asof)  # noqa: E731
    carry, nc, c = g("debt_carry"), g("debt_nc"), g("debt_c")
    comp = None
    if nc:
        cc = c[1] if (c and abs((c[0] - nc[0]).days) < 100) else 0.0
        comp = (nc[0], nc[1] + cc)
    if carry and (not comp or carry[0] >= comp[0]):
        debt = carry
    else:
        debt = comp
    cash, sti, csti = g("cash"), g("st_inv"), g("cash_sti")
    if csti and (not cash or csti[0] >= cash[0]):
        cash_v = csti
    elif cash:
        cash_v = (cash[0], cash[1] + (sti[1] if sti and abs((sti[0] - cash[0]).days) < 100 else 0.0))
    else:
        cash_v = None
    ol, fl = g("op_lease"), g("fin_lease")
    due, sh = g("debt_due_12m"), g("shares_dil")
    dq = q[(q.ticker == tk) & (q.metric == "interest")]
    if asof is not None:
        dq = dq[dq.filed <= asof]
    dq = dq.sort_values("end").tail(4)
    intr = float(dq.value.sum()) if len(dq) == 4 else np.nan
    return {"debt": debt, "cash": cash_v, "op_lease": ol, "fin_lease": fl, "due12": due, "shares": sh, "interest_ttm": intr}


def latest_table() -> pd.DataFrame:
    q = _q()
    h = pd.read_parquet(DATA_DIR / "results" / "ai_history.parquet")
    L = pd.read_parquet(DATA_DIR / "results" / "ai_latest.parquet").set_index("ticker")
    rows = []
    for tk in NAMES:
        b = balance_sheet(q, tk)
        hh = h[h.ticker == tk].sort_values("period_end").iloc[-1]
        debt = b["debt"][1] if b["debt"] else np.nan
        cash = b["cash"][1] if b["cash"] else np.nan
        leases = (b["op_lease"][1] if b["op_lease"] else 0.0) + (b["fin_lease"][1] if b["fin_lease"] else 0.0)
        fcf = float(L.loc[tk, "fcf_ttm"])
        ocf = float(hh.ocf_ttm)
        ebit = float(hh.op_income_ttm)
        intr = b["interest_ttm"]
        due = b["due12"][1] if b["due12"] else 0.0
        need = max(0.0, -fcf) + due          # 향후 12개월 현금 필요: FCF 적자 + 원금 만기
        cq, oq = _latest(q, tk, "capex"), _latest(q, tk, "ocf")
        fcf_rr = 4 * (oq[1] - cq[1]) if cq and oq else np.nan      # 마지막 분기 속도가 12개월 이어질 때의 FCF
        need_rr = max(0.0, -fcf_rr) + due if fcf_rr == fcf_rr else np.nan
        rpo = L.loc[tk, "rpo"] if "rpo" in L else np.nan
        rows.append({"ticker": tk, "name": NAMES[tk], "debt": debt, "leases": leases, "cash": cash, "net_debt": debt + leases - cash,
                     "net_debt_ocf": (debt + leases - cash) / ocf if ocf > 0 else np.nan, "interest_ttm": intr,
                     "ebit_cov": ebit / intr if intr and intr > 0 else np.nan, "ocf_cov": (ocf + intr) / intr if intr and intr > 0 else np.nan,
                     "due12": due, "fcf_ttm": fcf, "need12": need, "runway_y": (cash / need) if need > 0 else np.inf,
                     "fcf_runrate": fcf_rr, "runway_rr_y": (cash / need_rr) if need_rr == need_rr and need_rr > 0 else np.inf,
                     "rpo": float(rpo) if rpo == rpo else np.nan, "rpo_debt": (float(rpo) / (debt + leases)) if rpo == rpo and (debt + leases) > 0 else np.nan,
                     "debt_asof": str(b["debt"][0].date()) if b["debt"] else "", "cash_asof": str(b["cash"][0].date()) if b["cash"] else "",
                     "lease_asof": str(b["op_lease"][0].date()) if b["op_lease"] else ""})
    return pd.DataFrame(rows)


# ── Merton ─────────────────────────────────────────────────────────────────
def merton(E, sigE, D, r, T=5.0):
    """주식가치 E·주식변동성 sigE·부채 D로 자산가치 V·자산변동성 sigV를 풀고, 5년 신용스프레드(bp)와 부도거리(d2)를 낸다."""
    if not (E > 0 and sigE > 0 and D > 0):
        return np.nan, np.nan, np.nan

    def eq(x):
        V, sV = np.exp(x[0]), np.exp(x[1])
        d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
        d2 = d1 - sV * np.sqrt(T)
        return [V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E, norm.cdf(d1) * sV * V - sigE * E]
    x0 = [np.log(E + D), np.log(max(sigE * E / (E + D), 0.05))]
    try:
        sol, info, ier, _ = fsolve(eq, x0, full_output=True)
    except Exception:
        return np.nan, np.nan, np.nan
    if ier != 1:
        return np.nan, np.nan, np.nan
    V, sV = np.exp(sol[0]), np.exp(sol[1])
    d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    d2 = d1 - sV * np.sqrt(T)
    pd_term = norm.cdf(d2) + (V / (D * np.exp(-r * T))) * norm.cdf(-d1)     # 위험 부채가치/무위험 부채가치
    spread = -np.log(max(pd_term, 1e-9)) / T
    return float(spread * 1e4), float(d2), float(D / V)


def merton_history(start="2024-06-01") -> pd.DataFrame:
    """월말 시점별 Merton 내재 스프레드(bp)·부도거리·자산대비 부채. 부채·주식수는 그 시점까지 공시된 값만 쓴다."""
    q = _q()
    r5 = pd.read_parquet(PARQUET / "fred_DGS5.parquet") if (PARQUET / "fred_DGS5.parquet").exists() else pd.read_parquet(PARQUET / "fred_DGS10.parquet")
    rf = r5.iloc[:, 0] / 100
    rows = []
    for tk in NAMES:
        f = PARQUET / _fname(tk)
        if not f.exists():
            continue
        px = pd.read_parquet(f)["close"].dropna()
        lr = np.log(px).diff()
        vol = lr.rolling(126, min_periods=60).std() * np.sqrt(252)
        for d in pd.date_range(max(pd.Timestamp(start), px.index[0] + pd.Timedelta(days=100)), px.index[-1], freq="ME"):
            d = px.index[px.index <= d][-1] if (px.index <= d).any() else None
            if d is None or np.isnan(vol.get(d, np.nan)):
                continue
            b = balance_sheet(q, tk, asof=d)
            if not b["debt"] or not b["shares"]:
                continue
            D = b["debt"][1] + (b["op_lease"][1] if b["op_lease"] else 0.0) + (b["fin_lease"][1] if b["fin_lease"] else 0.0)
            E = float(px.loc[d]) * b["shares"][1]
            r = float(rf[rf.index <= d].iloc[-1]) if (rf.index <= d).any() else 0.04
            s, d2, lev = merton(E, float(vol.loc[d]), D, r)
            rows.append({"date": d, "ticker": tk, "spread_bp": s, "dd": d2, "debt_to_assets": lev, "equity": E, "debt": D, "sig_e": float(vol.loc[d])})
    return pd.DataFrame(rows)


def merton_vs_issued(mh: pd.DataFrame, issuance: pd.DataFrame) -> pd.DataFrame:
    """Oracle·CoreWeave의 실제 신규 발행 조건과 같은 시점 Merton 값을 나란히 놓는다(표본이 적어 상관 대신 비교표).
    Oracle: 발행 스프레드(bp, USD 3~11년). CoreWeave: 쿠폰 − 같은 시점 UST 5년(bp)."""
    rows = []
    f5 = PARQUET / "fred_DGS5.parquet"
    r5 = pd.read_parquet(f5).iloc[:, 0] if f5.exists() else None
    for tk in BORROWERS:
        m = mh[mh.ticker == tk].set_index("date").spread_bp.dropna().sort_index()
        g = issuance[(issuance.ticker == tk) & (issuance.currency == "USD")].copy()
        if m.empty or g.empty:
            continue
        for d, x in g.groupby("filed"):
            mm = m[m.index <= d]
            if mm.empty or d < m.index[0]:
                continue
            if tk == "ORCL":
                y = x[(x.tenor >= 3) & (x.tenor <= 11)].spread_bp.mean()
            else:
                cpn = x.coupon.mean()
                rf = float(r5[r5.index <= d].iloc[-1]) if r5 is not None and (r5.index <= d).any() else np.nan
                y = (cpn - rf) * 100 if cpn == cpn else np.nan
            if y == y:
                rows.append({"ticker": tk, "date": d, "issued_bp": float(y), "merton_bp": float(mm.iloc[-1])})
    return pd.DataFrame(rows)


# ── 증분 ROIC ──────────────────────────────────────────────────────────────
def incremental_roic() -> pd.DataFrame:
    """2022년 대비 늘어난 CAPEX(누적) 대비 영업이익 증가. 영업이익 증가를 전부 추가 투자의 성과로 돌리는 낙관적 계산(광고·유통 성장 포함)이다."""
    from src.ai.forecast import big4_quarterly
    M = big4_quarterly()
    cap, op = M["capex"], M["op_income"]
    base_cap = cap[(cap.index >= "2022-01-01") & (cap.index <= "2022-12-31")].mean()
    base_op = op[(op.index >= "2022-01-01") & (op.index <= "2022-12-31")].sum()
    rows = []
    for end in cap[cap.index >= "2024-12-31"].index:
        ttm_op = op[(op.index > end - pd.DateOffset(years=1)) & (op.index <= end)].sum()
        cum = (cap[(cap.index >= "2023-01-01") & (cap.index <= end)] - base_cap).clip(lower=0).sum()
        cum_lag = (cap[(cap.index >= "2023-01-01") & (cap.index <= end - pd.DateOffset(months=12))] - base_cap).clip(lower=0).sum()
        d_nopat = (ttm_op - base_op) * (1 - TAX)
        rows.append({"end": end, "cum_excess_capex": cum, "cum_excess_capex_lag4": cum_lag, "d_opinc": ttm_op - base_op, "d_nopat": d_nopat,
                     "roic": d_nopat / cum if cum > 0 else np.nan, "roic_lag4": d_nopat / cum_lag if cum_lag > 0 else np.nan,
                     "needed_nopat": HURDLE * cum, "gap": d_nopat - HURDLE * cum})
    return pd.DataFrame(rows)


def roic_by_company() -> pd.DataFrame:
    """회사별 증분 ROIC. 2022년 분기 평균 CAPEX를 기준선으로, 그보다 더 쓴 누적 CAPEX 대비 2022년 대비 영업이익 증가(세후)."""
    q = _q()

    def qs(tk, m):
        g = q[(q.ticker == tk) & (q.metric == m)].copy()
        g["pe"] = pd.DatetimeIndex(g.end) + pd.offsets.QuarterEnd(0)
        return g.sort_values("filed").drop_duplicates("pe", keep="first").set_index("pe").value.sort_index()

    rows = []
    for tk in ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRWV"]:
        cap, op = qs(tk, "capex"), qs(tk, "op_income")
        c22 = cap[(cap.index >= "2022-01-01") & (cap.index <= "2022-12-31")]
        o22 = op[(op.index >= "2022-01-01") & (op.index <= "2022-12-31")]
        base = float(c22.mean()) if len(c22) == 4 else 0.0          # CoreWeave는 2022년 이전 기록이 없어 0
        base_op = float(o22.sum()) if len(o22) == 4 else 0.0
        end = cap.index[-1]
        cum = float((cap[(cap.index >= "2023-01-01") & (cap.index <= end)] - base).clip(lower=0).sum())
        cum4 = float((cap[(cap.index >= "2023-01-01") & (cap.index <= end - pd.DateOffset(months=12))] - base).clip(lower=0).sum())
        top = float(op[(op.index > end - pd.DateOffset(years=1)) & (op.index <= end)].sum())
        dn = (top - base_op) * (1 - TAX)
        rows.append({"ticker": tk, "name": NAMES[tk], "end": end, "cum_excess_capex": cum, "cum_lag4": cum4, "d_nopat": dn, "roic": dn / cum if cum > 0 else np.nan,
                     "roic_lag4": dn / cum4 if cum4 > 0 else np.nan, "op_ttm": top})
    return pd.DataFrame(rows)


def maturity_ladder() -> pd.DataFrame:
    """공시된 원금 상환 일정(10-K 기준 최신): 1년 이내·2~5년·5년 초과. 차입 의존 기업과 빅테크를 같이 본다."""
    q = _q()
    rows = []
    for tk in ["ORCL", "CRWV", "META", "AMZN", "GOOGL", "MSFT"]:
        vals = {}
        for k, m in [("1년 이내", "debt_due_12m"), ("2년차", "debt_y2"), ("3년차", "debt_y3"), ("4년차", "debt_y4"), ("5년차", "debt_y5"), ("5년 초과", "debt_after5")]:
            x = _latest(q, tk, m)
            vals[k] = (x[1] if x else np.nan, x[0] if x else None)
        if np.isfinite(vals["1년 이내"][0]):
            asof = vals["1년 이내"][1]
            for k, (v, e) in vals.items():
                rows.append({"ticker": tk, "name": NAMES[tk], "bucket": k, "amount": v, "asof": str(asof.date()), "stale": bool(e is not None and e < asof - pd.Timedelta(days=100))})
    return pd.DataFrame(rows)


def valuation() -> pd.DataFrame:
    """시장이 AI 수익을 얼마나 반영했나: 시가총액, 매출 배수(P/S), FCF 수익률, CAPEX/시가총액, 52주 고점 대비, 변동성. 가격×최근 희석 주식수."""
    q = _q()
    h = pd.read_parquet(DATA_DIR / "results" / "ai_history.parquet")
    rows = []
    for tk in NAMES:
        f = PARQUET / _fname(tk)
        sh = _latest(q, tk, "shares_dil")
        if not f.exists() or not sh:
            continue
        px = pd.read_parquet(f)["close"].dropna()
        hh = h[h.ticker == tk].sort_values("period_end").iloc[-1]
        mc = float(px.iloc[-1]) * sh[1]
        b = balance_sheet(q, tk)
        netdebt = (b["debt"][1] if b["debt"] else 0.0) - (b["cash"][1] if b["cash"] else 0.0)
        fcf = float(hh.fcf_ttm)
        rows.append({"ticker": tk, "name": NAMES[tk], "mcap": mc, "ev": mc + netdebt, "ps": mc / float(hh.revenue_ttm), "fcf_yield": fcf / mc,
                     "capex_mcap": float(hh.capex_ttm) / mc, "dd52": float(px.iloc[-1] / px.tail(252).max() - 1),
                     "vol": float(np.log(px).diff().tail(126).std() * np.sqrt(252)), "asof": str(px.index[-1].date())})
    return pd.DataFrame(rows)


def oas_now() -> pd.DataFrame:
    rows = []
    for sid, nm in {"BAMLC0A4CBBB": "BBB", "BAMLH0A1HYBB": "BB", "BAMLH0A2HYB": "B", "BAMLH0A3HYC": "CCC"}.items():
        f = PARQUET / f"fred_{sid}.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f).iloc[:, 0].dropna()
        rows.append({"bucket": nm, "now": float(s.iloc[-1]) * 100, "asof": str(s.index[-1].date()), "chg_3m": float(s.iloc[-1] - s.iloc[-64]) * 100 if len(s) > 64 else np.nan,
                     "pct": float((s <= s.iloc[-1]).mean() * 100), "min3y": float(s.min()) * 100, "max3y": float(s.max()) * 100})
    return pd.DataFrame(rows)


def flags(lt: pd.DataFrame, mh: pd.DataFrame, ro: pd.DataFrame, rc: pd.DataFrame = None) -> list:
    out = []
    b = lt.set_index("ticker")
    # F11 유동성 런웨이: 현금 / (FCF 적자 + 12개월 만기)
    run = {t: float(b.loc[t, "runway_y"]) for t in BORROWERS if t in b.index}
    mn = min(run.values()) if run else np.inf
    lvl = "경고" if mn < 1 else ("주의" if mn < 2 else "정상")
    txt = ", ".join(f"{NAMES[t]} {v:.1f}년" if np.isfinite(v) else f"{NAMES[t]} 제한 없음" for t, v in run.items())
    out.append({"id": "F11", "level": lvl, "title": "차입 의존 기업의 12개월 유동성 런웨이", "text": f"현금 ÷ (FCF 적자 + 12개월 내 원금 만기): {txt}. 신규 조달 없이 버틸 수 있는 기간으로, 1년 미만이면 경고(미사용 한도·계약 자금은 반영 안 됨)."})
    # F12 이자보상배율
    cov = {t: float(b.loc[t, "ebit_cov"]) for t in BORROWERS if t in b.index and np.isfinite(b.loc[t, "ebit_cov"])}
    if cov:
        mc = min(cov.values())
        lvl = "경고" if mc < 1.5 else ("주의" if mc < 4 else "정상")
        out.append({"id": "F12", "level": lvl, "title": "이자보상배율(영업이익 ÷ 이자비용)", "text": ", ".join(f"{NAMES[t]} {v:.1f}배" for t, v in cov.items()) + " — 4배 미만이면 주의, 1.5배 미만이면 경고(영업적자면 음수)."})
    # F13 Merton 스트레스: 최신 내재 스프레드의 자기 이력 백분위
    if len(mh):
        pct = {}
        for t in BORROWERS:
            g = mh[mh.ticker == t].dropna(subset=["spread_bp"]).sort_values("date")
            if len(g) >= 6:
                pct[t] = (float(g.spread_bp.iloc[-1]), float((g.spread_bp <= g.spread_bp.iloc[-1]).mean() * 100))
        if pct:
            worst = max(p for _, p in pct.values())
            lvl = "주의" if worst >= 80 else ("중립" if worst >= 50 else "정상")
            out.append({"id": "F13", "level": lvl, "title": "주가 기반 신용 스트레스(Merton, CDS 대용)", "text": ", ".join(f"{NAMES[t]} {s:.0f}bp(자기 이력 {p:.0f}백분위)" for t, (s, p) in pct.items()) + " — CDS 시세가 아니라 주가·변동성·부채로 계산한 추정이며 수준보다 변화 방향을 봅니다."})
    # F14 증분 ROIC: Big4 합산은 맥락, 판정은 차입 의존 기업(Oracle·CoreWeave) 기준
    if rc is not None and len(rc):
        r = rc.set_index("ticker")
        bad = {t: float(r.loc[t, "roic"]) for t in BORROWERS if t in r.index and np.isfinite(r.loc[t, "roic"])}
        big = ro.iloc[-1] if ro is not None and len(ro) else None
        worst = min(bad.values()) if bad else np.nan
        lvl = "경고" if worst < 0.05 else ("주의" if worst < HURDLE else "양호")
        txt = ", ".join(f"{NAMES[t]} {v*100:.0f}%" for t, v in bad.items())
        if big is not None:
            txt += f" · Big4 합산 {big.roic*100:.0f}%"
        out.append({"id": "F14", "level": lvl, "title": "증분 ROIC (2022년 대비 늘린 CAPEX의 수익)", "text": txt + f" — 세후 영업이익 증가 ÷ 2022년보다 더 쓴 누적 CAPEX. 자본비용 가정 {HURDLE*100:.0f}% 미만이면 주의. 영업이익 증가 전부를 이 투자의 성과로 보는 낙관적 계산이고, 투자 직후에는 수익이 아직 안 나와 낮게 나옵니다."})
    return out


def run():
    lt = latest_table()
    mh = merton_history()
    from src.ai import credit as CR
    val = merton_vs_issued(mh, CR.issuance()) if (PARQUET / "debt_issuance.parquet").exists() else pd.DataFrame()
    ro = incremental_roic()
    rc = roic_by_company()
    oas = oas_now()
    return lt, mh, val, ro, rc, oas, flags(lt, mh, ro, rc), maturity_ladder(), valuation()
