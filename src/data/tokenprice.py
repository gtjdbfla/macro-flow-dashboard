"""AI 모델 토큰 가격 — OpenRouter 공개 모델 목록(/api/v1/models)의 게시가격.

OpenRouter를 거치는 API 가격이라 시장 일부이고 정가다(대량 계약 할인 제외). 과거 가격 API가 없어 수집할 때마다 (fetched, id) 행으로 누적한다.
단위: USD / 100만 토큰. 무료(0)·가변(-1) 가격과 ':free'·':batch' 같은 변형 id는 분석에서 뺀다."""
import pandas as pd
import requests

from . import PARQUET

URL = "https://openrouter.ai/api/v1/models"


def collect() -> dict:
    r = requests.get(URL, headers={"User-Agent": "macro-flow-dashboard research gtjdbfla@gmail.com"}, timeout=60)
    r.raise_for_status()
    rows = []
    for m in r.json()["data"]:
        p = m.get("pricing") or {}
        rows.append({"id": m["id"], "created": pd.to_datetime(m.get("created"), unit="s"),
                     "prompt": float(p.get("prompt") or 0) * 1e6, "completion": float(p.get("completion") or 0) * 1e6})
    new = pd.DataFrame(rows)
    new["fetched"] = pd.Timestamp.now().normalize()
    f = PARQUET / "token_prices.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    df = pd.concat([old, new]).drop_duplicates(["fetched", "id"], keep="last").sort_values(["fetched", "id"])
    df.to_parquet(f)
    return {"days": int(df.fetched.nunique()), "latest": str(df.fetched.max().date()), "models": len(new)}
