"""FRED API (FRED_API_KEY 환경변수). 월/분기 지표는 vintage 전체를 받아 available_at(=realtime_start)을 보존한다."""
import os
import time

import pandas as pd
import requests

from . import PARQUET

BASE = "https://api.stlouisfed.org/fred/series/observations"


def _get(params, tries=4):
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY 없음")
    for i in range(tries):
        try:
            r = requests.get(BASE, params={**params, "api_key": key, "file_type": "json"}, timeout=60)
            r.raise_for_status()
            return r.json()["observations"]
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def daily(series_id: str) -> pd.DataFrame:
    """현재 값만. 일별 시장 시리즈는 개정이 거의 없다(주간 WALCL 등 공표 시차는 feature 단계에서 반영)."""
    df = pd.DataFrame(_get({"series_id": series_id}))[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")


def vintage(series_id: str) -> pd.DataFrame:
    """관측일(date)별 공표시점(available_at)과 개정치 전부."""
    obs = _get({"series_id": series_id, "realtime_start": "1991-01-01", "realtime_end": "9999-12-31"})
    df = pd.DataFrame(obs)[["date", "realtime_start", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["available_at"] = pd.to_datetime(df.pop("realtime_start"))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values(["date", "available_at"])


def collect(daily_ids, vintage_ids) -> dict:
    out = {}
    for s in daily_ids:
        try:
            df = daily(s)
            df.to_parquet(PARQUET / f"fred_{s}.parquet")
            out[s] = {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": len(df)}
        except Exception as e:
            out[s] = {"error": str(e)[:200]}
    for s in vintage_ids:
        try:
            df = vintage(s)
            df.to_parquet(PARQUET / f"fredv_{s}.parquet")
            out[s] = {"start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
                      "n": len(df), "vintage": True}
        except Exception as e:
            out[s] = {"error": str(e)[:200]}
    return out
