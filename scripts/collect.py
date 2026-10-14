"""수집 진입점. 인자로 소스 지정(yahoo cftc fred), 없으면 전부. 요약은 data/coverage.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DATA_DIR, cftc, cloud, debt, fred, sec, ssga, twse, yahoo  # noqa: E402
from src.data.universe import CFTC, FRED_DAILY, FRED_VINTAGE, KR_SECTOR_STOCKS, YAHOO  # noqa: E402

only = sys.argv[1:]
cov_path = DATA_DIR / "coverage.json"
cov = json.loads(cov_path.read_text(encoding="utf-8")) if cov_path.exists() else {}
now = datetime.now(timezone.utc).isoformat(timespec="seconds")

if not only or "yahoo" in only:
    cov["yahoo"] = {"updated": now, "series": yahoo.collect(YAHOO)}
if not only or "cftc" in only:
    try:
        cov["cftc"] = {"updated": now, "series": cftc.collect(CFTC)}
    except Exception as e:
        cov["cftc"] = {"updated": now, "error": str(e)[:200], "series": {}}
if not only or "fred" in only:
    try:
        cov["fred"] = {"updated": now, "series": fred.collect(FRED_DAILY, FRED_VINTAGE)}
    except Exception as e:
        cov["fred"] = {"updated": now, "error": str(e)[:200], "series": {}}

if not only or "kis" in only:
    try:
        from src.data import kis
        cov["kis"] = {"updated": now, "series": kis.collect()}
    except Exception as e:
        cov["kis"] = {"updated": now, "error": str(e)[:200], "series": {}}

if not only or "kis_stocks" in only:
    try:
        from src.data import kis
        codes = [c for g in KR_SECTOR_STOCKS.values() for c in g]
        cov["kis_stocks"] = {"updated": now, "series": kis.collect_stocks(codes)}
    except Exception as e:
        cov["kis_stocks"] = {"updated": now, "error": str(e)[:200], "series": {}}

if not only or "sec" in only:
    cov["sec"] = {"updated": now, "series": sec.collect()}
if not only or "cloud" in only:
    cov["cloud"] = {"updated": now, "series": cloud.collect()}
if not only or "debt" in only:
    cov["debt"] = {"updated": now, "series": debt.collect()}
if not only or "twse" in only:
    try:
        cov["twse"] = {"updated": now, "series": {"snapshot": twse.collect()}}
    except Exception as e:
        cov["twse"] = {"updated": now, "series": {}, "error": str(e)[:200]}
if not only or "ssga" in only:
    cov["ssga"] = {"updated": now, "series": ssga.collect()}

cov_path.write_text(json.dumps(cov, ensure_ascii=False, indent=1), encoding="utf-8")
for src, v in cov.items():
    s = v.get("series", {})
    print(src, "ok", sum("error" not in x for x in s.values()), "err",
          sum("error" in x for x in s.values()), v.get("error", ""))
