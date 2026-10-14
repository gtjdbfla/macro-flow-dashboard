"""SPDR(State Street) ETF 일별 NAV·발행좌수(Shares Outstanding)·순자산 — 설정/환매 자금유입 계산의 원천.
무료·공식 발행사 파일(ssga.com navhist xlsx), ETF 설정일부터 일별. 발행좌수 변화 x NAV = 순 설정(유입) 금액.
iShares/VanEck/Global X 계열(SOXX, SMH, IGV, IBB, ITA, OIH 등)은 같은 형태의 무료 이력 파일을 찾지 못해 제외."""
import io
import time

import pandas as pd
import requests

from . import PARQUET

URL = "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/navhist-us-en-{t}.xlsx"
TICKERS = ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLB", "XLU", "XLRE", "SPY",
           "XSD", "KBE", "KIE", "XOP", "XPH", "XME", "XBI", "XAR", "KRE", "XHB", "XRT"]


def fetch(ticker: str) -> pd.DataFrame:
    last = None
    for i in range(3):
        try:
            r = requests.get(URL.format(t=ticker.lower()), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
            d = pd.read_excel(io.BytesIO(r.content), header=None, skiprows=4, usecols=[0, 1, 2, 3],
                              names=["date", "nav", "shares", "tna"])
            d["date"] = pd.to_datetime(d["date"], format="%d-%b-%Y", errors="coerce")
            d = d.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
            return d.apply(pd.to_numeric, errors="coerce").dropna(subset=["shares", "nav"])
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"ssga {ticker}: {last}")


def collect(tickers=TICKERS) -> dict:
    out = {}
    for t in tickers:
        try:
            d = fetch(t)
            d.to_parquet(PARQUET / f"ssga_{t}.parquet")
            out[t] = {"start": str(d.index[0].date()), "end": str(d.index[-1].date()), "n": len(d)}
        except Exception as e:
            out[t] = {"error": str(e)[:200]}
        time.sleep(0.5)
    return out


HOLD_URL = "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{t}.xlsx"


def fetch_holdings(ticker: str) -> pd.DataFrame:
    """ETF 보유 종목(이름·티커·비중%). 비중이 큰 순서. 현금·선물 등 티커 없는 줄은 뺀다."""
    r = requests.get(HOLD_URL.format(t=ticker.lower()), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), header=None)
    asof = pd.to_datetime(str(raw.iloc[2, 1]).replace("As of", "").strip(), format="%d-%b-%Y", errors="coerce")
    d = raw.iloc[5:, [0, 1, 4]].copy()
    d.columns = ["name", "ticker", "weight"]
    d["weight"] = pd.to_numeric(d["weight"], errors="coerce")
    d = d.dropna(subset=["weight", "ticker"])
    d = d[d.ticker.astype(str).str.fullmatch(r"[A-Z.\-]{1,6}")]
    d = d.sort_values("weight", ascending=False).reset_index(drop=True)
    d.insert(0, "etf", ticker)
    d["as_of"] = asof
    d["rank"] = range(1, len(d) + 1)
    return d


def collect_holdings(tickers=TICKERS) -> dict:
    parts, out = [], {}
    for t in tickers:
        if t == "SPY":
            continue
        try:
            d = fetch_holdings(t)
            parts.append(d)
            out[t] = {"n": len(d), "as_of": str(d.as_of.iloc[0].date())}
        except Exception as e:
            out[t] = {"error": str(e)[:200]}
        time.sleep(0.5)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(PARQUET / "ssga_holdings.parquet")
    return out
