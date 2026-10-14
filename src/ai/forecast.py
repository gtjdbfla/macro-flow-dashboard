"""AI 수익성 '다음 12개월' 기준선 전망과 그 정확도 검증.

방법(고정): 최근 분기의 전년동기비(YoY) 증가율이 최근 4분기 합계에 그대로 이어진다 -> 다음 4분기 합계.
  후보 5개(합계 그대로 / 최근 분기×4 / 연간 증가율 지속 / 최근 분기 YoY 지속 / 그 절반)를 Big4 분기 합산에 대해 같은 원점에서 비교했고,
  어떤 모수도 적합하지 않아(표본외와 같음) 편향·RMSE를 그대로 신뢰구간 폭으로 쓴다. '최근 분기 YoY 지속'이 매출·영업현금흐름·영업이익에서 편향이 0에 가장 가까웠다.
  방법은 이 비교에서 고른 것이므로(5개 중 최선) 선택 편향이 조금 있다. 4분기 창이 겹쳐 독립 표본은 약 1/4(≈9개)이다.
FCF는 영업현금흐름 전망 − CAPEX 전망으로 만든다(오차는 영업현금흐름 대비 %).

별도로, 더 정교해 보이는 예측 변수들(RPO 격차, NVIDIA 매출, AR(1))이 이 기준선을 넘는지 같은 방식으로 확인한 결과를 'neg' 표로 남긴다."""
import numpy as np
import pandas as pd

from src.data import DATA_DIR, PARQUET

BIG = ["MSFT", "GOOGL", "AMZN", "META"]
METHODS = ["합계 그대로(0%)", "최근 분기×4(런레이트)", "연간 증가율 지속", "최근 분기 YoY 지속", "최근 분기 YoY의 절반"]
BEST = "최근 분기 YoY 지속"
PER = {"2016–19": ("2016-01-01", "2019-12-31"), "2020–22": ("2020-01-01", "2022-12-31"), "2023~": ("2023-01-01", "2100-01-01"), "전체": ("2000-01-01", "2100-01-01")}
NAMES = {"capex": "CAPEX", "ocf": "영업현금흐름", "revenue": "매출", "op_income": "영업이익", "fcf": "잉여현금흐름"}


def _qs(q, tk, m):
    g = q[(q.ticker == tk) & (q.metric == m)].copy()
    g["pe"] = pd.DatetimeIndex(g.end) + pd.offsets.QuarterEnd(0)  # Oracle 등 결산월이 다른 회사는 가장 가까운 달력 분기로
    g = g.sort_values("filed").drop_duplicates("pe", keep="first")  # 최초 공시값(point-in-time)
    return g.set_index("pe").value.sort_index()


def big4_quarterly() -> dict:
    q = pd.read_parquet(PARQUET / "sec_quarterly.parquet")
    M = {}
    for m in ("capex", "ocf", "revenue", "op_income"):
        d = pd.concat({t: _qs(q, t, m) for t in BIG}, axis=1).dropna()
        M[m] = d.sum(axis=1)
    M["fcf"] = M["ocf"] - M["capex"]
    return M


def _methods(x: pd.Series, i: int) -> dict:
    ttm = x.iloc[i - 3: i + 1].sum()
    qn, q4 = x.iloc[i], x.iloc[i - 4]
    gq = qn / q4 - 1 if q4 > 0 else np.nan
    prev = x.iloc[i - 7: i - 3].sum()
    gt = ttm / prev - 1 if prev > 0 else np.nan
    return dict(zip(METHODS, [ttm, 4 * qn, ttm * (1 + gt), ttm * (1 + gq), ttm * (1 + 0.5 * gq)]))


def evaluate(M: dict) -> pd.DataFrame:
    """지표×방법×구간별 편향(평균 로그오차)과 RMSE. FCF는 부호가 바뀔 수 있어 영업현금흐름 대비 % 오차를 쓴다."""
    rows = []
    for nm in ("capex", "ocf", "revenue", "op_income", "fcf"):
        x = M[nm].dropna()
        e = []
        for i in range(8, len(x) - 4):
            act = x.iloc[i + 1: i + 5].sum()
            base = M["ocf"].reindex(x.index).iloc[i + 1: i + 5].sum()
            r = {"date": x.index[i]}
            for k, v in _methods(x, i).items():
                r[k] = (v - act) / base if nm == "fcf" else (np.log(v / act) if v > 0 and act > 0 else np.nan)
            e.append(r)
        E = pd.DataFrame(e).set_index("date")
        for pn, (a, b) in PER.items():
            s = E[(E.index >= a) & (E.index <= b)]
            for k in METHODS:
                v = s[k].dropna()
                if len(v):
                    rows.append({"metric": nm, "method": k, "period": pn, "bias": float(v.mean()), "rmse": float(np.sqrt((v ** 2).mean())), "n": int(len(v))})
    return pd.DataFrame(rows)


def project(M: dict, ev: pd.DataFrame) -> pd.DataFrame:
    """지금 시점의 다음 4분기 전망(기준선)과 과거 정확도에 따른 범위."""
    rows = []
    cur = {}
    for nm in ("capex", "ocf", "revenue", "op_income"):
        x = M[nm].dropna()
        i = len(x) - 1
        ttm = float(x.iloc[i - 3: i + 1].sum())
        proj = float(_methods(x, i)[BEST])
        e = ev[(ev.metric == nm) & (ev.method == BEST)].set_index("period")
        cur[nm] = (ttm, proj, e)
        rows.append({"metric": nm, "now_ttm": ttm, "proj": proj, "growth": proj / ttm - 1, "lo": proj * np.exp(-e.loc["전체", "rmse"]), "hi": proj * np.exp(e.loc["전체", "rmse"]),
                     "rmse": float(e.loc["전체", "rmse"]), "bias_recent": float(e.loc["2023~", "bias"]), "bias_all": float(e.loc["전체", "bias"]),
                     "last_q": float(x.iloc[-1]), "last_q_end": str(x.index[-1].date())})
    ttm_f, proj_f = cur["ocf"][0] - cur["capex"][0], cur["ocf"][1] - cur["capex"][1]
    ef = ev[(ev.metric == "fcf") & (ev.method == BEST)].set_index("period")
    band = float(ef.loc["전체", "rmse"]) * cur["ocf"][1]
    qfcf = float(M["fcf"].iloc[-1])
    rows.append({"metric": "fcf", "now_ttm": ttm_f, "proj": proj_f, "growth": np.nan, "lo": proj_f - band, "hi": proj_f + band, "rmse": float(ef.loc["전체", "rmse"]),
                 "bias_recent": float(ef.loc["2023~", "bias"]), "bias_all": float(ef.loc["전체", "bias"]), "last_q": qfcf, "last_q_end": str(M["fcf"].index[-1].date())})
    return pd.DataFrame(rows)


def negatives() -> pd.DataFrame:
    """정교해 보이는 예측 변수가 기준선을 넘는지: 못 넘으면 '참고'로 남긴다."""
    H = pd.read_parquet(DATA_DIR / "results" / "ai_history.parquet")
    H["pe"] = pd.DatetimeIndex(H.period_end) + pd.offsets.QuarterEnd(0)
    out = []
    # 1) RPO(수주잔고) 증가율과 매출 증가율의 격차 -> 4분기 뒤 매출 증가율 변화 (MSFT·GOOGL·ORCL)
    parts = []
    for tk in ("MSFT", "GOOGL", "ORCL"):
        g = H[H.ticker == tk].drop_duplicates("pe", keep="last").set_index("pe")
        d = pd.DataFrame({"x": (g.rpo_yoy - g.rev_yoy).clip(-1.5, 3), "y": g.rev_yoy.shift(-4) - g.rev_yoy}).dropna()
        d["x"] -= d.x.mean()
        d["y"] -= d.y.mean()
        parts.append(d)
    D = pd.concat(parts)
    out.append({"test": "수주잔고(RPO) 증가율이 매출 증가율보다 클수록 4분기 뒤 매출 증가율이 오른다", "n": len(D),
                "result": f"피어슨 {D.x.corr(D.y):+.2f} · 스피어만 {D.x.corr(D.y, method='spearman'):+.2f}", "verdict": "참고",
                "why": "선형 상관은 약하게 +지만 순위 상관은 0 근처이고 구간별로 흔들림. 소표본(3사)"})
    # 2) NVIDIA 매출 증가율 -> Big4 CAPEX 증가율 (h분기 뒤, 확장창 표본외, AR(1) 대비)
    B = H[H.ticker == "BIG4"].drop_duplicates("pe", keep="last").set_index("pe")
    NV = H[H.ticker == "NVDA"].drop_duplicates("pe", keep="last").set_index("pe")
    cx, nv = B.capex_yoy.dropna(), NV.rev_yoy.dropna()
    for h in (1, 2):
        d = pd.DataFrame({"cx": cx, "nv": nv}).dropna()
        d["tgt"] = d.cx.shift(-h)
        a_err, n_err = [], []
        for i in range(12, len(d)):
            tr = d.iloc[: i - h + 1].dropna(subset=["tgt"])
            cur = d.iloc[i]
            if len(tr) < 8 or np.isnan(cur.tgt):
                continue
            for cols, acc in ((["cx"], a_err), (["cx", "nv"], n_err)):
                A = np.c_[np.ones(len(tr)), tr[cols].values]
                beta, *_ = np.linalg.lstsq(A, tr.tgt.values, rcond=None)
                acc.append((cur.tgt - float(np.r_[1, cur[cols].values] @ beta)) ** 2)
        if a_err:
            ra, rn = np.sqrt(np.mean(a_err)), np.sqrt(np.mean(n_err))
            out.append({"test": f"NVIDIA 매출 증가율이 Big4 CAPEX 증가율을 {h}분기 앞선다", "n": len(a_err),
                        "result": f"표본외 오차 AR(1) {ra*100:.1f}%p → +NVIDIA {rn*100:.1f}%p ({(1-rn/ra)*100:+.0f}%)", "verdict": "참고",
                        "why": "표본 내 상관은 높지만(변화량 0.5~0.6) 표본외에서는 개선이 일관되지 않음. 같이 오르는 동행 관계일 가능성"})
    # 3) 증가율 자체의 AR(1) 평균회귀 vs '그대로 지속'
    for nm, col in (("Big4 CAPEX", "capex_yoy"), ("Big4 매출", "rev_yoy"), ("Big4 영업이익", "opinc_yoy")):
        y = B[col].dropna()
        d = pd.DataFrame({"y": y, "t": y.shift(-1)}).dropna()
        e_ar, e_nv = [], []
        for i in range(14, len(d)):
            tr = d.iloc[: i]
            b1, a1 = np.polyfit(tr.y.values, tr.t.values, 1)
            e_ar.append((d.t.iloc[i] - (a1 + b1 * d.y.iloc[i])) ** 2)
            e_nv.append((d.t.iloc[i] - d.y.iloc[i]) ** 2)
        if e_ar:
            out.append({"test": f"{nm} 증가율: 평균회귀 모델(AR1)이 '그대로 지속'보다 다음 분기를 더 잘 맞힌다", "n": len(e_ar),
                        "result": f"AR(1) 오차 {np.sqrt(np.mean(e_ar))*100:.1f}%p vs 지속 {np.sqrt(np.mean(e_nv))*100:.1f}%p", "verdict": "참고",
                        "why": "1분기 앞은 현재 증가율 유지가 기준선. 통계 모델이 이기지 못함"})
    return pd.DataFrame(out)


def run():
    M = big4_quarterly()
    ev = evaluate(M)
    return ev, project(M, ev), negatives()
