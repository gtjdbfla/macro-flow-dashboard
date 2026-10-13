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
