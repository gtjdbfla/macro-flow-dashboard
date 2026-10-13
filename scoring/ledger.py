"""전향 검증 장부(append-only). 오늘 이후의 실제 데이터로만 가설을 평가한다 — 과거 재구성(백필) 금지.

기록(매일 1행, as_of 기준 중복 기록 안 함): 한국 외국인 수급 예측 z, VIX 순위, 한국 업종별 외국인 60일 순매수 비율, 레짐.
평가는 실현값이 생긴 행만 사용. 판정 규칙은 docs/FORWARD_TEST.md에 사전 고정(커밋 시점이 사전 등록 시각)."""
import json

import numpy as np
import pandas as pd

from backtest.forward_returns import Fwd
from src.data import DATA_DIR, PARQUET
from src.features.common import Ctx

LEDGER = DATA_DIR / "ledger" / "ledger.jsonl"
KR_ETF = {"Semis": "091160.KS", "Auto": "091180.KS", "Banks": "091170.KS", "Healthcare": "143860.KS",
          "EnergyChem": "117460.KS", "Battery": "305720.KS", "Shipbuilding": "466920.KS", "Defense": "463250.KS"}


from src.features.sector_features import SECTORS as US_SECTORS  # noqa: E402


def _read():
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]


def record(summary: dict) -> bool:
    """as_of가 이미 있으면 기록하지 않는다. 새로 기록했으면 True."""
    rows = _read()
    asof = summary["as_of"]
    if any(r["date"] == asof for r in rows):
        return False
    risk = pd.read_parquet(PARQUET / "features_risk.parquet")
    kor = pd.read_parquet(PARQUET / "features_korea.parquet")
    state = pd.read_parquet(PARQUET / "features_state.parquet")
    d = pd.Timestamp(asof)
    kh = pd.read_parquet(DATA_DIR / "results" / "korea_model_history.parquet")
    sec = pd.read_parquet(PARQUET / "features_sector.parquet")
    rec = {"date": asof, "korea_pred_z": float(kh.loc[d, "pred_foreign_z"]) if d in kh.index else None,
           "vix_rank": _f(risk.loc[d, "risk__VIX__level_rank"]), "regime": state.loc[d, "regime"],
           "kr_foreign_ratio_60d": {s: _f(kor.loc[d, f"kr__{s}_foreign__ratio_60d"]) for s in KR_ETF},
           "us_flow_20d": {t: _f(sec.loc[d, f"sector__{lab}__flow_20d"]) for t, lab in US_SECTORS.items()}}
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def _f(v):
    return None if v is None or v != v else float(v)


def evaluate() -> dict:
    rows = _read()
    out = {"started": rows[0]["date"] if rows else None, "n_days": len(rows)}
    if not rows:
        return out
    ctx = Ctx()
    fw = Fwd(ctx)
    idx = ctx.cal
    pos = {d: i for i, d in enumerate(idx)}
    # H1 한국 모델: 예측 z vs 실제 다음 한국 거래일 z
    kh = pd.read_parquet(DATA_DIR / "results" / "korea_model_history.parquet")
    pairs = [(r["korea_pred_z"], kh["actual_next_foreign_z"].get(pd.Timestamp(r["date"]))) for r in rows
             if r["korea_pred_z"] is not None]
    pairs = [(a, b) for a, b in pairs if b is not None and b == b]
    out["h1_korea"] = {"n": len(pairs), "corr": float(np.corrcoef(*zip(*pairs))[0, 1]) if len(pairs) >= 20 else None,
                       "sign_hit": float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs])) if pairs else None}
    # H2 VIX 역방향: 고공포(순위≥0.8) 시점 이후 SPY 20일 수익률 vs 무조건 중앙값
    y, _ = fw.abs("SPY", 20)
    hi = [y[pos[pd.Timestamp(r["date"])]] for r in rows if r["vix_rank"] is not None and r["vix_rank"] >= 0.8
          and pd.Timestamp(r["date"]) in pos]
    hi = [v for v in hi if v == v]
    base = np.nanmedian(y[:-20 - 130]) if len(y) > 200 else np.nan
    out["h2_vix"] = {"n_high_fear": len(hi), "median_after": float(np.median(hi)) if hi else None,
                     "baseline_median": float(base), "n_days_recorded": len(rows)}
    # H3 한국 업종 외국인 60일 비율 -> 이후 20일 업종 ETF의 KOSPI 대비 수익 (단면 Spearman IC)
    Y = {s: fw.rel(e, "^KS11", 20) for s, e in KR_ETF.items()}
    ics = []
    for r in rows:
        d = pd.Timestamp(r["date"])
        if d not in pos:
            continue
        x, yy = [], []
        for s in KR_ETF:
            v = r["kr_foreign_ratio_60d"].get(s)
            if v is not None and Y[s] is not None and np.isfinite(Y[s][0][pos[d]]):
                x.append(v)
                yy.append(Y[s][0][pos[d]])
        if len(x) >= 5:
            ics.append(pd.Series(x).rank().corr(pd.Series(yy).rank()))
    out["h3_kr_sector"] = {"n_matured": len(ics), "ic_mean": float(np.mean(ics)) if ics else None}
    # H4 미국 섹터 20일 순유입(%AUM)이 클수록 이후 20일 SPY 대비 수익이 낮다(역방향) -> IC<0 기대
    ys = {t: fw.rel(t, "SPY", 20) for t in US_SECTORS}
    ics4 = []
    for r in rows:
        d = pd.Timestamp(r["date"])
        if d not in pos or "us_flow_20d" not in r:
            continue
        x, yy = [], []
        for t in US_SECTORS:
            v = r["us_flow_20d"].get(t)
            if v is not None and ys[t] is not None and np.isfinite(ys[t][0][pos[d]]):
                x.append(v)
                yy.append(ys[t][0][pos[d]])
        if len(x) >= 8:
            ics4.append(pd.Series(x).rank().corr(pd.Series(yy).rank()))
    out["h4_us_flow"] = {"n_matured": len(ics4), "ic_mean": float(np.mean(ics4)) if ics4 else None}
    return out
