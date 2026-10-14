"""대만 AI 공급망 월매출 — TWSE 공개 API(t187ap05_L, 상장사 최신월 월매출). 월 10일경 전월 값이 공개되는 선행 지표.

API는 '최신월 스냅샷'만 주므로 수집할 때마다 (회사, 월) 행으로 누적하고, 스냅샷에 들어 있는 '전년 동월 매출'도 함께 저장한다.
따라서 이력은 이 프로젝트가 수집을 시작한 시점부터 쌓인다(과거 백필 없음). 매출 단위: 천 NT$."""
import pandas as pd
import requests

from . import PARQUET

URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
AI_CHAIN = {"2330": "TSMC", "2317": "Foxconn(鴻海)", "2382": "Quanta(廣達)", "6669": "Wiwynn(緯穎)", "3231": "Wistron(緯創)",
            "2356": "Inventec(英業達)", "2345": "Accton(智邦)", "3711": "ASE(日月光)", "2376": "Gigabyte(技嘉)",
            "3017": "AVC(奇鋐)", "2308": "Delta(台達電)", "2454": "MediaTek(聯發科)"}
ROLE = {"2330": "파운드리", "2317": "AI 서버 조립", "2382": "AI 서버 ODM", "6669": "AI 서버(하이퍼스케일러 직납)",
        "3231": "AI 서버 ODM", "2356": "서버 ODM", "2345": "AI 네트워크 스위치", "3711": "패키징·테스트",
        "2376": "AI 서버·메인보드", "3017": "서버 냉각", "2308": "전원·냉각", "2454": "팹리스(AI ASIC)"}
F = {"ym": "資料年月", "rev": "營業收入-當月營收", "rev_prior_month": "營業收入-上月營收", "rev_prior_year": "營業收入-去年當月營收",
     "mom": "營業收入-上月比較增減(%)", "yoy": "營業收入-去年同月增減(%)", "ytd": "累計營業收入-當月累計營收",
     "ytd_yoy": "累計營業收入-前期比較增減(%)"}


def _month(ym: str) -> pd.Timestamp:
    return pd.Timestamp(year=int(ym[:-2]) + 1911, month=int(ym[-2:]), day=1) + pd.offsets.MonthEnd(0)


def collect() -> dict:
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    rows = []
    for x in r.json():
        c = x.get("公司代號")
        if c in AI_CHAIN:
            rec = {"ticker": c, "name": AI_CHAIN[c], "month": _month(x[F["ym"]]), "fetched": pd.Timestamp.now().normalize()}
            for k, f in F.items():
                if k != "ym":
                    try:
                        rec[k] = float(x[f])
                    except (TypeError, ValueError):
                        rec[k] = float("nan")
            rows.append(rec)
    new = pd.DataFrame(rows)
    f = PARQUET / "twse_monthly.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    df = pd.concat([old, new]).sort_values("fetched").drop_duplicates(["ticker", "month"], keep="last").sort_values(["ticker", "month"])
    df.to_parquet(f)
    return {"months": int(df.month.nunique()), "latest": str(df.month.max().date()), "tickers": int(df.ticker.nunique())}
