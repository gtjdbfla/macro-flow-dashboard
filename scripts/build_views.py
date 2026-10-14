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
