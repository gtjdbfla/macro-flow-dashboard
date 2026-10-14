"""Ken French Data Library(다트머스) — 1926~ 월별 49개 산업 수익률(가치가중)과 팩터(시장·SMB·HML·RMW·CMA·모멘텀).
무료·공개 파일. 가격 기반이라 '자금 유입' 근거는 아니고, 산업·팩터 모멘텀을 100년 표본으로 검정하는 데 쓴다.
매월 중순에 지난달까지로 갱신된다(발표 지연 약 1~1.5개월)."""
import io
import zipfile

import pandas as pd
import requests

from . import PARQUET

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"ind": ("49_Industry_Portfolios_CSV.zip", "49_Industry_Portfolios.csv"),
         "ff": ("F-F_Research_Data_5_Factors_2x3_CSV.zip", "F-F_Research_Data_5_Factors_2x3.csv"),
         "mom": ("F-F_Momentum_Factor_CSV.zip", "F-F_Momentum_Factor.csv")}


def _is_month(line: str) -> bool:
    head = line.split(",")[0].strip()
    return len(head) == 6 and head.isdigit()


def parse_first_monthly_block(text: str) -> pd.DataFrame:
    """파일 맨 앞 월별 블록(YYYYMM 행)만 읽는다. 수익률은 % -> 소수. -99.99/-999는 결측."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if _is_month(ln))
    hdr = [c.strip() for c in lines[i - 1].split(",")][1:]
    rows = []
    while i < len(lines) and _is_month(lines[i]):
        rows.append(lines[i].split(","))
        i += 1
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df[0].str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.iloc[:, 1:1 + len(hdr)].apply(pd.to_numeric, errors="coerce")
    df.columns = hdr
    return df.where(df > -99) / 100


def collect() -> dict:
    out = {}
    for key, (zname, member) in FILES.items():
        r = requests.get(BASE + zname, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
        r.raise_for_status()
        text = zipfile.ZipFile(io.BytesIO(r.content)).read(member).decode("latin-1")
        df = parse_first_monthly_block(text)
        df.index.name = "date"
        df.to_parquet(PARQUET / f"french_{key}.parquet")
        out[key] = {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": len(df)}
    return out
