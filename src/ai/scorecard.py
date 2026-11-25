"""전향 성적표(docs/AI_PROFITABILITY.md 2026-11-25 사전 등록). 가이드 장부의 날짜별 신호를 이후 실제 결과로 채점한다.
결과: 스냅샷 날짜 종가 대비 이후 63거래일 안에 QQQ가 −10% 이상 빠졌는가. 63거래일이 지난 날만 확정으로 센다."""
import pandas as pd

HORIZON, DROP, MIN_DAYS = 63, -0.10, 60
STAGE_RANK = {"정상": 1, "회복 관찰": 1, "주의": 2, "경계": 3, "위기": 4}


def outcomes(rows, price):
    """확정된 스냅샷 날짜별 결과(True면 −10% 발생). 미확정 날짜는 빠진다."""
    out = {}
    for r in rows:
        d = pd.Timestamp(r["date"])
        past = price.loc[:d]
        if past.empty:
            continue
        after = price.loc[d:].iloc[1:]
        if len(after) < HORIZON:
            continue
        out[r["date"]] = bool(after.iloc[:HORIZON].min() / past.iloc[-1] - 1 <= DROP)
    return out


def signals(r):
    e = r.get("early") or {}
    s = {f"{k} 빨강": v == "red" for k, v in e.get("levels", {}).items()}
    rank = STAGE_RANK.get(e.get("stage"), 0)
    if e.get("stage"):
        s["단계 주의 이상"] = rank >= 2
        s["단계 경계 이상"] = rank >= 3
    for t, name in (("^GSPC", "S&P 500"), ("QQQ", "QQQ"), ("SOXX", "SOXX")):
        if t in e.get("trend", {}):
            s[f"{name} 이탈"] = e["trend"][t] == "이탈"
    return s


def score(rows, price):
    rows = [r for r in rows if r.get("early")]
    res = outcomes(rows, price)
    base = sum(res.values()) / len(res) if res else None
    table = {}
    for r in rows:
        if r["date"] not in res:
            continue
        for k, on in signals(r).items():
            t = table.setdefault(k, {"on": 0, "hit": 0})
            if on:
                t["on"] += 1
                t["hit"] += res[r["date"]]
    for t in table.values():
        t["rate"] = t["hit"] / t["on"] if t["on"] else None
    return {"snapshots": len(rows), "matured": len(res), "base_rate": base, "enough": len(res) >= MIN_DAYS,
            "first_snapshot": rows[0]["date"] if rows else None, "signals": table}
