"""GPU 임대가격 — 06.ai 공개 CSV(34개 공급업체 게시가격의 절사평균, 1달러=GPU 1시간).

게시 가격(온디맨드·스팟·1M~3Y 약정 호가)이며 실제 장기계약 체결가와 다르다(체결가는 SemiAnalysis 등 유료 자료).
일별 CSV가 공개된 구간이 짧아(2026-10-26~) 수집할 때마다 (day, index) 행으로 누적한다. 단위: USD / GPU-시간."""
import io

import pandas as pd
import requests

from . import PARQUET

URL = "https://06.ai/data/gpu-index-daily.csv"


def collect() -> dict:
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    new = pd.read_csv(io.StringIO(r.text), parse_dates=["day"])
    new["fetched"] = pd.Timestamp.now().normalize()
    f = PARQUET / "gpu_rental.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    df = pd.concat([old, new]).sort_values("fetched").drop_duplicates(["day", "index"], keep="last").sort_values(["index", "day"])
    df.to_parquet(f)
    return {"days": int(df.day.nunique()), "latest": str(df.day.max().date()), "indices": int(df["index"].nunique())}
