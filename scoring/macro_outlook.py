"""미국 주식 투자 환경을 설명하는 현재 시점의 매크로 점검표."""
import numpy as np
import pandas as pd

from src.data import PARQUET


# 변화폭 기준은 방향을 읽기 위한 운영 규칙이다. 수익률로 최적화하지 않았다.
METRICS = {
    "rates": ("10년 실질금리", "20거래일", "bp", 25),
    "credit": ("하이일드 신용 스프레드", "20거래일", "bp", 20),
    "liquidity": ("연준 순유동성", "4주", "%", 1),
    "labor": ("신규 실업수당 청구 4주 평균", "4주", "%", 5),
    "inflation": ("근원 CPI 전년비", "3개월", "%p", 0.2),
}


def _fred(sid, as_of):
    path = PARQUET / f"fred_{sid}.parquet"
    if not path.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.read_parquet(path)["value"].dropna().sort_index()
    return s.loc[:as_of - pd.Timedelta(days=1)]


def _first_print(sid, as_of):
    path = PARQUET / f"fredv_{sid}.parquet"
    if not path.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    d = pd.read_parquet(path)
    return (d.loc[(d.available_at <= as_of) & (d.date <= as_of)]
            .sort_values("available_at").drop_duplicates("date").set_index("date")["value"].sort_index())


def _metric(key, s, change, as_of, max_age):
    _, _, _, threshold = METRICS[key]
    if s.empty:
        return {"state": "missing", "as_of": None, "value": None, "change": None}
    last = s.index[-1]
    value, delta = s.iloc[-1], change.iloc[-1]
    if (as_of - last).days > max_age:
        state = "stale"
    elif not (np.isfinite(value) and np.isfinite(delta)):
        state = "missing"
    else:
        pressure = -delta if key == "liquidity" else delta
        state = "adverse" if pressure >= threshold else "supportive" if pressure <= -threshold else "neutral"
    return {"state": state, "as_of": last.date().isoformat(),
            "value": float(value) if np.isfinite(value) else None,
            "change": float(delta) if np.isfinite(delta) else None}


def outlook(metrics, market_level, shock, market_ready):
    states = [metric["state"] for metric in metrics.values()]
    if not market_ready or any(s in {"missing", "stale"} for s in states):
        return "pending"
    adverse = states.count("adverse")
    stress = shock or market_level in {"높음", "매우 높음"}
    if adverse >= 3 or (adverse >= 2 and stress):
        return "caution"
    if states.count("supportive") >= 3 and adverse == 0 and not stress:
        return "supportive"
    return "mixed"


def snapshot(as_of, market_level, shock, market_ready):
    as_of = pd.Timestamp(as_of)
    rates = _fred("DFII10", as_of)
    credit = _fred("BAMLH0A0HYM2", as_of)
    # 수요일 기준 Fed 자산·TGA와 같은 날짜의 역레포만 결합한다. 단위는 십억 달러.
    liquidity = ((_fred("WALCL", as_of) - _fred("WTREGEN", as_of)) / 1000
                 - _fred("RRPONTSYD", as_of)).dropna().asfreq("W-WED")
    claims = _first_print("ICSA", as_of).asfreq("W-SAT").rolling(4).mean()
    cpi = _first_print("CPILFESL", as_of).asfreq("MS").pct_change(12, fill_method=None) * 100
    metrics = {
        "rates": _metric("rates", rates, rates.diff(20).mul(100).round(6), as_of, 5),
        "credit": _metric("credit", credit, credit.diff(20).mul(100).round(6), as_of, 5),
        "liquidity": _metric("liquidity", liquidity, liquidity.pct_change(4, fill_method=None) * 100, as_of, 10),
        "labor": _metric("labor", claims, claims.pct_change(4, fill_method=None) * 100, as_of, 14),
        "inflation": _metric("inflation", cpi, cpi.diff(3), as_of, 80),
    }
    return {"as_of": as_of.date().isoformat(), "metrics": metrics,
            "outlook": outlook(metrics, market_level, shock, market_ready)}
