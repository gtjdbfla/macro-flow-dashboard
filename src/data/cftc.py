"""CFTC TFF (Futures Only). 보고 기준일=화요일, 공표=금요일 -> available_at = 기준일+3일."""
import pandas as pd
import requests

from . import PARQUET

URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
COLS = ["report_date_as_yyyy_mm_dd", "cftc_contract_market_code", "open_interest_all",
        "dealer_positions_long_all", "dealer_positions_short_all",
        "asset_mgr_positions_long", "asset_mgr_positions_short",
        "lev_money_positions_long", "lev_money_positions_short",
        "other_rept_positions_long", "other_rept_positions_short"]


def collect(codes) -> dict:
    where = "cftc_contract_market_code in (" + ",".join(f"'{c}'" for c in codes) + ")"
    rows, offset = [], 0
    while True:
        r = requests.get(URL, params={"$select": ",".join(COLS), "$where": where,
                                      "$limit": 50000, "$offset": offset,
                                      "$order": "report_date_as_yyyy_mm_dd"}, timeout=90)
        r.raise_for_status()
        chunk = r.json()
        rows += chunk
        if len(chunk) < 50000:
            break
        offset += 50000
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.pop("report_date_as_yyyy_mm_dd")).dt.normalize()
    df["available_at"] = df["date"] + pd.Timedelta(days=3)
    for c in df.columns:
        if c not in ("date", "available_at", "cftc_contract_market_code"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(["date", "cftc_contract_market_code"]).sort_values("date")
    df.to_parquet(PARQUET / "cftc_tff.parquet")
    return {code: {"start": str(g["date"].min().date()), "end": str(g["date"].max().date()), "n": len(g)}
            for code, g in df.groupby("cftc_contract_market_code")}
