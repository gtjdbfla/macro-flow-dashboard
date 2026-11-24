"""Cboe 일간 풋콜 비율. 2006-11~2019-10은 공개 CSV(주식·전체), 이후는 일별 JSON을 날짜마다 받아 이어붙인다.
없는 날(휴장일)은 404라 건너뛴다. 한 번 받은 날은 다시 받지 않는다(누적). 단위: 비율(풋/콜)."""
import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import PARQUET

UA = {"User-Agent": "Mozilla/5.0 macro-flow-dashboard research gtjdbfla@gmail.com"}
CSV = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{}.csv"
DAILY = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{}_daily_options"
CSV_END = pd.Timestamp("2019-10-04")
FILE = PARQUET / "cboe_putcall.parquet"


def _csv(name):
    r = requests.get(CSV.format(name), headers=UA, timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    start = next(i for i, x in enumerate(lines) if x.startswith("DATE"))
    d = pd.read_csv(io.StringIO("\n".join(lines[start:])), skipinitialspace=True)
    d["date"] = pd.to_datetime(d["DATE"], format="mixed")
    return d.set_index("date")["P/C Ratio"].astype(float)


def _day(d):
    r = requests.get(DAILY.format(d.strftime("%Y-%m-%d")), headers=UA, timeout=30)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    ratios = {x["name"]: x["value"] for x in r.json().get("ratios", [])}
    if "EQUITY PUT/CALL RATIO" not in ratios:
        return None
    return {"date": d, "equity": float(ratios["EQUITY PUT/CALL RATIO"]), "total": float(ratios["TOTAL PUT/CALL RATIO"])}


def collect() -> dict:
    old = pd.read_parquet(FILE) if FILE.exists() else pd.DataFrame(columns=["equity", "total"], index=pd.DatetimeIndex([], name="date"))
    if old.empty:
        hist = pd.DataFrame({"equity": _csv("equitypc"), "total": _csv("totalpc")})
        hist = hist[hist.index <= CSV_END]
        old = hist
    last = old.index.max()
    days = [d for d in pd.bdate_range(last + pd.Timedelta(days=1), pd.Timestamp.now().normalize())]
    with ThreadPoolExecutor(8) as ex:
        got = [x for x in ex.map(lambda d: _try(d), days) if x]
    new = pd.DataFrame(got).set_index("date") if got else pd.DataFrame(columns=["equity", "total"])
    df = pd.concat([old, new]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(FILE)
    return {"rows": len(df), "latest": str(df.index.max().date()), "new": len(new)}


def _try(d):
    for _ in range(3):
        try:
            return _day(d)
        except Exception:
            continue
    return None
