import re
import time

import pandas as pd
import requests

from . import PARQUET

UA = {"User-Agent": "Mozilla/5.0"}


def _fname(symbol: str) -> str:
    return "px_" + re.sub(r"[^A-Za-z0-9]", "_", symbol) + ".parquet"


def fetch(symbol: str) -> pd.DataFrame:
    # range=max는 종목에 따라 월봉으로 내려온다(2026-10-13 발견). period1=0으로 일봉을 강제한다.
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params={"period1": 0, "period2": int(time.time()) + 86400, "interval": "1d"}, headers=UA, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            break
        except Exception as e:  # 비공식 API: 재시도 후 실패 보고
            last = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"yahoo {symbol}: {last}")
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q["close"], "adjclose": adj or q["close"], "volume": q["volume"],
    }).dropna(subset=["close"]).drop_duplicates("date", keep="last")
    return df.set_index("date")


def collect(symbols) -> dict:
    out = {}
    for s in symbols:
        try:
            df = fetch(s)
            df.to_parquet(PARQUET / _fname(s))
            out[s] = {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": len(df)}
        except Exception as e:
            out[s] = {"error": str(e)[:200]}
        time.sleep(0.3)
    return out
