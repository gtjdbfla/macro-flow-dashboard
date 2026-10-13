"""한국투자증권 Open API — 시장별 투자자 매매동향(일별). KOSPI/KOSDAQ 외국인·기관·개인 순매수.

실측(2026-10-13): 1회 300거래일, DATE_1에서 과거로 거슬러 올라가며 2002년까지 응답한다.
금액 단위는 백만원, 수량 단위는 천주. 값은 장 마감 후 확정(당일 장중 값은 잠정).

토큰은 stock_dashboard와 같은 앱키를 쓰므로 그쪽 캐시를 읽기전용으로 공유한다(KIS_TOKEN_FILE).
토큰을 발급할 때마다 카카오 알림톡이 가므로 만료 전에는 절대 새로 받지 않는다.
"""
import datetime as dt
import json
import os
import time

import pandas as pd
import requests

from . import DATA_DIR, PARQUET

BASE = os.environ.get("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
SHARED_TOKEN = os.environ.get("KIS_TOKEN_FILE", "")
OWN_TOKEN = DATA_DIR / "kis_token.json"
MARKETS = {"KOSPI": ("0001", "KSP"), "KOSDAQ": ("1001", "KSQ")}
KEEP = ["bstp_nmix_prpr", "frgn_ntby_qty", "prsn_ntby_qty", "orgn_ntby_qty",
        "frgn_ntby_tr_pbmn", "prsn_ntby_tr_pbmn", "orgn_ntby_tr_pbmn",
        "scrt_ntby_tr_pbmn", "ivtr_ntby_tr_pbmn", "pe_fund_ntby_tr_pbmn", "bank_ntby_tr_pbmn",
        "insu_ntby_tr_pbmn", "fund_ntby_tr_pbmn", "etc_orgt_ntby_tr_pbmn", "etc_corp_ntby_tr_pbmn"]


def _valid(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
        exp = dt.datetime.fromisoformat(d["expires_at"])
        if dt.datetime.now(dt.timezone.utc) < exp - dt.timedelta(minutes=10):
            return d["access_token"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def token() -> str:
    for p in (SHARED_TOKEN, OWN_TOKEN):
        if p and (t := _valid(p)):
            return t
    r = requests.post(f"{BASE}/oauth2/tokenP", timeout=15, json={
        "grant_type": "client_credentials",
        "appkey": os.environ["KIS_APP_KEY"], "appsecret": os.environ["KIS_APP_SECRET"]})
    r.raise_for_status()
    b = r.json()
    exp = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=int(b.get("expires_in", 86400)))
    OWN_TOKEN.write_text(json.dumps({"access_token": b["access_token"], "expires_at": exp.isoformat()}))
    return b["access_token"]


def _page(tok, code, sub, date_str):
    r = requests.get(
        f"{BASE}/uapi/domestic-stock/v1/quotations/inquire-investor-daily-by-market",
        headers={"authorization": f"Bearer {tok}", "appkey": os.environ["KIS_APP_KEY"],
                 "appsecret": os.environ["KIS_APP_SECRET"], "tr_id": "FHPTJ04040000", "custtype": "P"},
        params={"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": date_str,
                "FID_INPUT_ISCD_1": sub, "FID_INPUT_DATE_2": date_str, "FID_INPUT_ISCD_2": code},
        timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("rt_cd") != "0":
        raise RuntimeError(j.get("msg1"))
    return j.get("output") or []


def fetch_market(name: str, since="2002-01-01") -> pd.DataFrame:
    code, sub = MARKETS[name]
    tok = token()
    cursor = dt.date.today()
    frames = []
    while cursor > dt.date.fromisoformat(since):
        rows = _page(tok, code, sub, cursor.strftime("%Y%m%d"))
        if not rows:
            break
        df = pd.DataFrame(rows)
        frames.append(df)
        oldest = dt.datetime.strptime(df["stck_bsop_date"].min(), "%Y%m%d").date()
        if oldest >= cursor:
            break
        cursor = oldest - dt.timedelta(days=1)
        time.sleep(0.25)  # 초당 호출 제한 여유
    df = pd.concat(frames).drop_duplicates("stck_bsop_date")
    df["date"] = pd.to_datetime(df["stck_bsop_date"])
    cols = [c for c in KEEP if c in df.columns]
    out = df.set_index("date")[cols].apply(pd.to_numeric, errors="coerce").sort_index()
    return out


def collect() -> dict:
    res = {}
    for m in MARKETS:
        try:
            df = fetch_market(m)
            df.to_parquet(PARQUET / f"kis_investor_{m.lower()}.parquet")
            res[m] = {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": len(df)}
        except Exception as e:
            res[m] = {"error": str(e)[:200]}
    return res


STOCK_COLS = ["stck_clpr", "acml_vol", "acml_tr_pbmn", "frgn_ntby_qty", "orgn_ntby_qty", "prsn_ntby_qty",
              "frgn_ntby_tr_pbmn", "orgn_ntby_tr_pbmn", "prsn_ntby_tr_pbmn"]


def _stock_page(tok, code, date_str):
    r = requests.get(
        f"{BASE}/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
        headers={"authorization": f"Bearer {tok}", "appkey": os.environ["KIS_APP_KEY"],
                 "appsecret": os.environ["KIS_APP_SECRET"], "tr_id": "FHPTJ04160001", "custtype": "P"},
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": date_str,
                "FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("rt_cd") != "0":
        raise RuntimeError(j.get("msg1"))
    return j.get("output2") or j.get("output") or []


def fetch_stock(code: str, since="2005-01-01", have_until=None) -> pd.DataFrame:
    """종목별 일별 투자자 수급(30행/회, 과거로 페이징). have_until이 있으면 그 날짜까지만(증분)."""
    tok = token()
    floor = dt.date.fromisoformat(since)
    if have_until is not None:
        floor = max(floor, have_until - dt.timedelta(days=7))  # 최근 며칠은 확정값으로 덮어쓴다
    cursor, frames = dt.date.today(), []
    while cursor > floor:
        try:
            rows = _stock_page(tok, code, cursor.strftime("%Y%m%d"))
        except Exception:
            time.sleep(2)
            try:
                rows = _stock_page(tok, code, cursor.strftime("%Y%m%d"))
            except Exception:
                if frames:  # 상장 이전 구간은 500을 내는 경우가 있다 -> 거기까지를 이력으로 본다
                    break
                raise
        if not rows:
            break
        df = pd.DataFrame(rows)
        frames.append(df)
        oldest = dt.datetime.strptime(df["stck_bsop_date"].min(), "%Y%m%d").date()
        if oldest >= cursor:
            break
        cursor = oldest - dt.timedelta(days=1)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=STOCK_COLS)
    df = pd.concat(frames).drop_duplicates("stck_bsop_date")
    df["date"] = pd.to_datetime(df["stck_bsop_date"])
    out = df.set_index("date")[[c for c in STOCK_COLS if c in df.columns]].apply(pd.to_numeric, errors="coerce").sort_index()
    return out[out["acml_tr_pbmn"] > 0]  # 상장 이전 자리표시 행(NaN/0)과 거래정지일 제거


def collect_stocks(codes) -> dict:
    res = {}
    for code in codes:
        f = PARQUET / f"kis_stock_{code}.parquet"
        try:
            old = pd.read_parquet(f) if f.exists() else None
            new = fetch_stock(code, have_until=None if old is None else old.index.max().date())
            df = new if old is None else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            df.to_parquet(f)
            res[code] = {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": len(df)}
        except Exception as e:
            res[code] = {"error": str(e)[:200]}
    return res
