"""대시보드용 요약 파일 생성 -> data/views/.  앱은 67MB feature 파일을 열지 않고 이 작은 파일들(+results)만 읽는다.
실행 순서: collect -> build_features -> run_backtests -> run_ai -> run_scoring -> build_views"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import DATA_DIR, PARQUET  # noqa: E402
from src.features.sector_features import SECTORS  # noqa: E402

R = DATA_DIR / "results"
V = DATA_DIR / "views"
V.mkdir(exist_ok=True)
F = lambda n: pd.read_parquet(PARQUET / f"features_{n}.parquet")  # noqa: E731
summary = json.loads((R / "summary.json").read_text(encoding="utf-8"))

# ── 매크로 상태 ─────────────────────────────────────────────────────────────
st = F("state").dropna(subset=["state_key"])
cur = st.iloc[-1]
runs = st["regime"].ne(st["regime"].shift()).cumsum()
state = {"as_of": str(st.index[-1].date()), **{k: cur[k] for k in ["regime", "growth", "inflation", "policy", "liquidity", "dollar"]},
         "days_in_regime": int((runs == runs.iloc[-1]).sum()),
         "flips_1y": int(st["regime"].tail(252).ne(st["regime"].tail(252).shift()).sum() - 1)}
(V / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

# ── 섹터 ETF 순유입 이력(주간) + 레짐 ───────────────────────────────────────
sec = F("sector")
cols = {lab: f"sector__{lab}__flow_20d" for lab in SECTORS.values() if f"sector__{lab}__flow_20d" in sec}
fh = sec[list(cols.values())].rename(columns={v: k for k, v in cols.items()}).iloc[::5].loc["2007-06-01":]
fh["regime"] = st["regime"].reindex(fh.index)
fh.astype({c: "float32" for c in cols}).to_parquet(V / "flow_history.parquet")

# ── 기관 선물 포지션(S&P 섹터 E-mini) ───────────────────────────────────────
fl = F("flow").dropna(how="all")
last = fl.iloc[-1]
keys = {"S&P 500(전체)": "SP500_E-mini", "나스닥100": "Nasdaq100_E-mini", "Technology": "SP_Tech_E-mini", "Financials": "SP_Fin_E-mini",
        "Energy": "SP_Energy_E-mini", "Healthcare": "SP_Health_E-mini", "Industrials": "SP_Ind_E-mini", "Utilities": "SP_Util_E-mini",
        "ConsDisc": "SP_ConsDisc_E-mini", "ConsStaples": "SP_ConsStaples_E-mini", "Materials": "SP_Materials_E-mini"}
rows = []
for name, k in keys.items():
    g = lambda f: last.get(f"cftc__{k}__{f}", np.nan)  # noqa: E731
    rows.append({"name": name, "am_net": g("assetmgr_net") * 100, "am_rank": g("assetmgr_net_rank") * 100,
                 "am_chg4w": g("assetmgr_net_chg4w") * 100, "lev_net": g("levmoney_net") * 100, "lev_rank": g("levmoney_net_rank") * 100})
pd.DataFrame(rows).to_parquet(V / "cftc_now.parquet")

# ── 한국: 투자자 누적, 업종 흐름, 모델 5분위 ────────────────────────────────
kis = pd.read_parquet(PARQUET / "kis_investor_kospi.parquet").tail(260)
kis[["frgn_ntby_tr_pbmn", "orgn_ntby_tr_pbmn", "prsn_ntby_tr_pbmn"]].to_parquet(V / "kr_investor.parquet")
kr = F("korea").dropna(how="all")
krl = kr.iloc[-1]
names = {"Semis": "반도체", "Auto": "자동차", "Banks": "은행", "Healthcare": "헬스케어", "EnergyChem": "에너지화학", "Battery": "2차전지",
         "Shipbuilding": "조선", "Defense": "방산"}
pd.DataFrame([{"sector": v, "foreign_5d": krl.get(f"kr__{k}_foreign__ratio_5d", np.nan) * 100,
               "foreign_20d": krl.get(f"kr__{k}_foreign__ratio_20d", np.nan) * 100,
               "foreign_60d": krl.get(f"kr__{k}_foreign__ratio_60d", np.nan) * 100,
               "inst_20d": krl.get(f"kr__{k}_inst__ratio_20d", np.nan) * 100} for k, v in names.items()]).to_parquet(V / "kr_sector_now.parquet")

kh = pd.read_parquet(R / "korea_model_history.parquet")
px = pd.read_parquet(PARQUET / "px__KS11.parquet")
krd = px.index
rr = pd.DataFrame({"gap": px.open / px.close.shift(1) - 1, "intraday": px.close / px.open - 1, "total": px.close / px.close.shift(1) - 1})
x = kh.dropna(subset=["pred_foreign_z", "actual_next_foreign_z"]).copy()
nxt = [krd[krd.searchsorted(d, side="right")] if krd.searchsorted(d, side="right") < len(krd) else pd.NaT for d in x.index]
x["kr_next"] = nxt
x = x.dropna(subset=["kr_next"]).join(rr, on="kr_next")
q = []
for name, (a, b) in {"fit": ("2005", "2022"), "oos": ("2023", "2100")}.items():
    y = x.loc[a:b].dropna(subset=["gap", "intraday"]).copy()
    y["q"] = pd.qcut(y.pred_foreign_z, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    t = y.groupby("q", observed=True).agg(days=("pred_foreign_z", "size"), buy_prob=("actual_next_foreign_z", lambda s: (s > 0).mean() * 100),
                                          total_bp=("total", lambda s: s.mean() * 1e4), gap_bp=("gap", lambda s: s.mean() * 1e4),
                                          intraday_bp=("intraday", lambda s: s.mean() * 1e4)).reset_index()
    t["period"] = name
    q.append(t)
pd.concat(q).to_parquet(V / "korea_quintile.parquet")
kh.tail(120).to_parquet(V / "korea_model_recent.parquet")

# ── 점수 이력(주간) ─────────────────────────────────────────────────────────
pd.read_parquet(R / "scores_history.parquet").iloc[::5].loc["2015-01-01":].to_parquet(V / "scores_weekly.parquet")

# ── 근거 페이지용 수치 ─────────────────────────────────────────────────────
rep = pd.read_parquet(R / "regime_replication.parquet")
t = rep[rep.testable]
sr = pd.read_parquet(R / "sector_rotation.parquet")
ev = {"regime_testable": int(len(t)), "regime_train_sig": int(t.train_sig.sum()), "regime_replicated": int(t.replicated.sum()),
      "regime_rate": float(t.replicated.mean()), "sector_ic_combos": int(sr[sr.regime == "ALL"].groupby(["feature", "h"]).ngroups),
      "korea_eval": summary["korea_model"]["eval"], "oos_ic": {k: v["test"]["ic"] for k, v in summary["oos_ic"].items()},
      "ledger": summary.get("ledger", {}), "score_components": int(len(pd.read_parquet(R / "score_validation.parquet"))),
      "score_included": int(pd.read_parquet(R / "score_validation.parquet").included.sum())}
for _n in ("evidence_tests", "evidence_conditions", "korea_model_v2"):
    if (R / f"{_n}.parquet").exists():
        pd.read_parquet(R / f"{_n}.parquet").to_parquet(V / f"{_n}.parquet")
if (R / "evidence_now.json").exists():
    (V / "evidence_now.json").write_text((R / "evidence_now.json").read_text(encoding="utf-8"), encoding="utf-8")
(V / "evidence.json").write_text(json.dumps(ev, ensure_ascii=False, default=float), encoding="utf-8")

# ── 데이터 신선도 ───────────────────────────────────────────────────────────
cov = json.loads((DATA_DIR / "coverage.json").read_text(encoding="utf-8"))
sec_q = pd.read_parquet(PARQUET / "sec_quarterly.parquet")
tw = pd.read_parquet(PARQUET / "twse_monthly.parquet")
meta = {"us_close": summary["as_of"], "kis_last": str(kis.index[-1].date()), "sec_filed": str(sec_q["filed"].max().date()),
        "sec_period": str(sec_q["end"].max().date()), "twse_month": str(tw.month.max().date()),
        "updated": {k: v.get("updated") for k, v in cov.items()}}
(V / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
print("views:", sorted(p.name for p in V.glob("*")), "| size KB:", sum(p.stat().st_size for p in V.glob("*")) // 1024)

# ── 갱신 현황: 소스별 신선도 + 실행 일정 ──────────────────────────────────────
import datetime as _dt  # noqa: E402

today = _dt.datetime.now(_dt.timezone.utc).date()


def _pq(name, col, how="max"):
    try:
        d = pd.read_parquet(PARQUET / name)
        v = d.index.max() if col == "index" else d[col].max()
        return pd.Timestamp(v).date()
    except Exception:
        return None


def _fred_end():
    ends = [x.get("end") for x in cov.get("fred", {}).get("series", {}).values() if x.get("end")]
    return _dt.date.fromisoformat(max(ends)) if ends else None


# (키, 내용, 갱신 주기, 허용 경과일, 데이터 최신일)
SRC = [
    ("yahoo", "가격: 미국·한국 지수·ETF·종목·환율", "매 영업일 · 미국 마감 후", 4, _pq("px_SPY.parquet", "index")),
    ("kis", "한국 시장 투자자 수급(외국인·기관·개인)", "매 영업일 · 한국 장 마감 후", 5, _pq("kis_investor_kospi.parquet", "index")),
    ("kis_stocks", "한국 업종 대표 종목 수급", "매 영업일 · 한국 장 마감 후", 5, _pq("kis_stock_005930.parquet", "index")),
    ("cftc", "CFTC 선물 포지션(기관)", "주 1회 · 화요일 기준, 금요일 공표", 11, _pq("cftc_tff.parquet", "date")),
    ("fred", "FRED 금리·유동성·거시", "일별~월별 시리즈 혼합", 5, _fred_end()),
    ("ssga", "SPDR ETF 설정·환매(발행좌수)", "매 영업일", 5, _pq("ssga_XLK.parquet", "index")),
    ("sec", "SEC 분기 재무(CAPEX·현금흐름)", "분기 · 실적 시즌(1·4·7·10월 말)", 100, _pq("sec_quarterly.parquet", "filed")),
    ("cloud", "클라우드 부문·AI 수치(8-K 발표문)", "분기 · 실적 발표일", 100, _pq("cloud_segments.parquet", "filed")),
    ("debt", "채권 신규 발행 조건(SEC)", "발행 시(불규칙)", 120, _pq("debt_issuance.parquet", "filed")),
    ("twse", "대만 AI 공급망 월매출", "월 1회 · 매월 10일경", 45, _pq("twse_monthly.parquet", "month")),
    ("french", "Ken French 산업·팩터 수익률(1926~, 가격 기반)", "월 1회 · 매월 중순(지난달까지)", 80, _pq("french_ind.parquet", "index")),
]
rows = []
for key, label, cadence, tol, end in SRC:
    c = cov.get(key, {})
    series = c.get("series", {})
    n_err = sum("error" in x for x in series.values())
    age = (today - end).days if end else None
    if c.get("error") or (series and n_err == len(series)):
        status, note = "오류", c.get("error") or "전 시리즈 오류"
    elif series and n_err / len(series) > 0.3:
        status, note = "일부 오류", f"시리즈 {n_err}/{len(series)} 오류"
    elif age is None:
        status, note = "데이터 없음", ""
    elif age > tol:
        status, note = "지연", f"허용 {tol}일 초과"
    else:
        status, note = "정상", f"시리즈 {len(series)}개" + (f", 오류 {n_err}" if n_err else "")
    rows.append({"source": key, "content": label, "cadence": cadence, "last_run_utc": (c.get("updated") or "")[:16].replace("T", " "),
                 "data_end": str(end) if end else "—", "age_days": age, "tolerance": tol, "status": status, "note": note})
pd.DataFrame(rows).to_parquet(V / "sources.parquet")


def _next(now, hh, mm, weekdays):
    for d in range(0, 9):
        t = (now + _dt.timedelta(days=d)).replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t > now and t.weekday() in weekdays:
            return t
    return None


def _next_filings(now):
    season = (1, 4, 7, 10)
    t = now.replace(minute=15, second=0, microsecond=0)
    for _ in range(24 * 120):
        t += _dt.timedelta(hours=1)
        if t > now and t.month in season and 15 <= t.day <= 31 and t.hour % 3 == 0:
            return t
    return None


utc = _dt.timezone.utc
nowu = _dt.datetime.now(utc)
MODES = [
    ("kr", "한국 수급 갱신 → 분석·화면 갱신", "월~금 16:40 KST", _next(nowu, 7, 40, {0, 1, 2, 3, 4})),
    ("us", "미국 가격·거시·ETF·공시·채권 갱신 → 분석·화면 갱신 (대만 월매출은 매월 10~12일)", "월~금 07:30 KST(미국 마감 후)", _next(nowu, 22, 30, {0, 1, 2, 3, 4})),
    ("weekly", "백테스트·검증 통계 재계산", "일요일 18:00 KST", _next(nowu, 9, 0, {6})),
    ("filings", "실적 시즌 새 공시 확인(있을 때만 분석 갱신)", "1·4·7·10월 15~31일, 3시간마다", _next_filings(nowu)),
]
sched = []
for mode, what, when, nxt in MODES:
    p = DATA_DIR / f"status_{mode}.json"
    last = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    sched.append({"mode": mode, "what": what, "when": when,
                  "next_kst": (nxt + _dt.timedelta(hours=9)).strftime("%m-%d %H:%M") if nxt else "—",
                  "last_kst": (pd.Timestamp(last["finished"]) + pd.Timedelta(hours=9)).strftime("%m-%d %H:%M") if last else "아직 없음",
                  "last_ok": (None if not last else bool(last["ok"])), "last_s": (None if not last else last["duration_s"]),
                  "last_warn": ("; ".join(last["warnings"])[:160] if last else "")})
(V / "schedule.json").write_text(json.dumps(sched, ensure_ascii=False), encoding="utf-8")
print("sources/schedule built:", [r["status"] for r in rows])

# ── 홈 상세: 섹터 자금(달러), 다음 자금 이동 후보, AI 투자 주체·변화량 ───────────
from src.features.sector_features import SECTORS as _SEC  # noqa: E402

rows = []
for sym, lab in _SEC.items():
    f = PARQUET / f"ssga_{sym}.parquet"
    if not f.exists():
        continue
    d = pd.read_parquet(f)
    fl = (d["shares"].diff() * d["nav"]).where(d["shares"].pct_change().abs() < 0.4, 0.0)
    rows.append({"sector": lab, "flow_20d_usd": float(fl.tail(20).sum()), "flow_60d_usd": float(fl.tail(60).sum()),
                 "tna": float(d["tna"].iloc[-1]), "as_of": str(d.index[-1].date())})
pd.DataFrame(rows).to_parquet(V / "sector_flow_usd.parquet")

# 섹터 ETF 대표 종목(상위 8): SPDR 공개 보유 종목 파일. 산업 ETF는 동일가중이라 '대표'가 의미 없어 제외
_hf = PARQUET / "ssga_holdings.parquet"
if _hf.exists():
    _h = pd.read_parquet(_hf)
    _h = _h[_h.etf.isin(list(_SEC))].copy()
    _h["sector"] = _h.etf.map(_SEC)
    _h["top_n"] = _h.groupby("etf").weight.transform("size")
    _h[_h["rank"] <= 8].to_parquet(V / "sector_holdings.parquet")

fr = pd.read_parquet(R / "flow_rules.parquet") if (R / "flow_rules.parquet").exists() else pd.DataFrame()
secn = pd.read_parquet(R / "sector_snapshot.parquet")
etf = secn[secn.flow_src == "ETF유입"].sort_values("flow_1m", ascending=False)
krn = pd.read_parquet(V / "kr_sector_now.parquet").sort_values("foreign_60d", ascending=False)
nxt = {"in_top": etf.head(3)[["name", "flow_1m"]].values.tolist(), "out_top": etf.tail(3).iloc[::-1][["name", "flow_1m"]].values.tolist(),
       "kr_in_top": krn.head(3)[["sector", "foreign_60d"]].values.tolist(), "kr_out_top": krn.tail(3).iloc[::-1][["sector", "foreign_60d"]].values.tolist(),
       "rules": fr.to_dict("records")}
(V / "flow_next.json").write_text(json.dumps(nxt, ensure_ascii=False, default=float), encoding="utf-8")

hist = pd.read_parquet(R / "ai_history.parquet")
cos = ["AMZN", "GOOGL", "MSFT", "META", "ORCL", "CRWV"]
NM = {"MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "ORCL": "Oracle", "CRWV": "CoreWeave"}
recs = []
for t in cos:
    h = hist[hist.ticker == t].dropna(subset=["capex_ttm"]).sort_values("period_end")
    if len(h) < 6:
        continue
    cur, y1, q2 = h.iloc[-1], h.iloc[-5], h.iloc[-3]
    recs.append({"ticker": t, "name": NM[t], "period_end": str(cur.period_end.date()), "capex": cur.capex_ttm, "capex_prev": y1.capex_ttm,
                 "capex_delta": cur.capex_ttm - y1.capex_ttm, "capex_yoy": cur.capex_yoy, "capex_yoy_2q": q2.capex_yoy,
                 "ocf": cur.ocf_ttm, "ocf_delta": cur.ocf_ttm - y1.ocf_ttm, "capex_ocf": cur.capex_ocf, "capex_ocf_2q": q2.capex_ocf,
                 "fcf": cur.fcf_ttm, "fcf_delta": cur.fcf_ttm - y1.fcf_ttm, "rev": cur.revenue_ttm, "rev_delta": cur.revenue_ttm - y1.revenue_ttm,
                 "op": cur.op_income_ttm, "op_delta": cur.op_income_ttm - y1.op_income_ttm, "rev_yoy": cur.rev_yoy, "rev_yoy_2q": q2.rev_yoy})
ha = pd.DataFrame(recs)
tot = ha.capex.sum()
ha["share"] = ha.capex / tot
ha["delta_share"] = ha.capex_delta / ha.capex_delta.sum()
ha.to_parquet(V / "home_ai.parquet")
sq = pd.read_parquet(PARQUET / "sec_quarterly.parquet")
cq = sq[(sq.metric == "capex") & sq.ticker.isin(cos)][["ticker", "end", "value"]].sort_values("end")
cq[cq.end >= cq.end.max() - pd.DateOffset(months=27)].to_parquet(V / "ai_capex_q.parquet")
print("home details built:", len(ha), "companies")
