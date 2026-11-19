"""보유 업종 점검 — 메모리·광통신·전력. 표시만 하며 판정·등급을 만들지 않는다(docs/AI_PROFITABILITY.md, 2026-11-19 사전 등록)."""
import pandas as pd

from src.ai.end_demand import basket
from src.data import PARQUET
from src.data.sec import MEMORY, OPTICAL, POWER
from src.data.yahoo import _fname

GROUPS = (("메모리", MEMORY), ("광통신", OPTICAL), ("전력 인프라", POWER))
NAMES = {"MU": "Micron", "LITE": "Lumentum", "COHR": "Coherent", "CIEN": "Ciena", "GLW": "Corning", "FN": "Fabrinet",
         "CRDO": "Credo", "ANET": "Arista", "VRT": "Vertiv", "GEV": "GE Vernova", "ETN": "Eaton", "PWR": "Quanta Services",
         "HUBB": "Hubbell", "NVDA": "NVIDIA", "000660.KS": "SK하이닉스"}


def _rpo_yoy(sec, ticker, as_of):
    d = sec[(sec.ticker == ticker) & (sec.metric == "rpo") & (sec.filed <= as_of)].sort_values("filed").drop_duplicates("end", keep="first")
    d = d.set_index("end").value.sort_index()
    if d.empty:
        return None
    last = d.index[-1]
    near = d.index[abs((d.index - (last - pd.DateOffset(years=1))).days) <= 10]
    return float(d.iloc[-1] / d[near[-1]] - 1) if len(near) else None


def fundamentals(sec, as_of=None):
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    out = []
    for group, tickers in GROUPS:
        t, _ = basket(sec, tickers, NAMES, as_of)
        t.insert(0, "group", group)
        t["rpo_yoy"] = [_rpo_yoy(sec, tk, as_of) for tk in t.ticker]
        out.append(t)
    return pd.concat(out, ignore_index=True)


def prices(as_of=None):
    """종가(수정주가 우선)의 52주 고점 대비 낙폭과 1·3개월 수익률."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    rows = []
    for ticker, name in NAMES.items():
        f = PARQUET / _fname(ticker)
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        s = (d["adjclose"] if "adjclose" in d else d["close"]).dropna()
        s = s[s.index <= as_of]
        if len(s) < 70:
            continue
        last = s.iloc[-1]
        ret = lambda days: float(last / s.iloc[-1 - days] - 1)  # noqa: E731  거래일 기준
        rows.append({"ticker": ticker, "name": name, "last_date": s.index[-1], "dd_52w": float(last / s.tail(252).max() - 1),
                     "ret_1m": ret(21), "ret_3m": ret(63)})
    return pd.DataFrame(rows)
