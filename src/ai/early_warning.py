"""조기경보 패널과 연속일 추세 상태판. 규칙·임계값은 docs/AI_PROFITABILITY.md(2026-11-24 사전 등록)에 있고 표시 전용이다.
Threads 게시물(@_dplabs_)의 체크리스트에서 우리 데이터로 계산 가능한 지표만 옮겼다. 임계값은 게시물의 값이며 검증되지 않았다."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.data.yahoo import _fname

K_TREND = 3
BREADTH_ETFS = ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLB", "XLU", "XLRE", "IWM", "QQQ", "SPY"]
TREND_ASSETS = (("S&P 500", "^GSPC", False), ("나스닥100(QQQ)", "QQQ", False), ("반도체(SOXX)", "SOXX", False),
                ("코스피", "^KS11", False), ("NVIDIA", "NVDA", True), ("SK하이닉스", "000660.KS", True),
                ("KODEX 미국AI광통신네트워크", "0173Y0.KS", True), ("AIPO", "AIPO", True), ("Bloom Energy", "BE", True),
                ("Fermi", "FRMI", True))
GREEN, YELLOW, RED, NA = "green", "yellow", "red", "na"


def _px(ticker):
    path = PARQUET / _fname(ticker)
    if not path.exists():
        return pd.Series(dtype=float)
    d = pd.read_parquet(path)
    return (d["adjclose"] if "adjclose" in d else d["close"]).dropna()


def _fred(sid):
    path = PARQUET / f"fred_{sid}.parquet"
    return pd.read_parquet(path)["value"].dropna() if path.exists() else pd.Series(dtype=float)


def _fredv(sid):
    """발표 시점 자료(fredv_)에서 관측월마다 가장 최근 값."""
    path = PARQUET / f"fredv_{sid}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    d = pd.read_parquet(path).sort_values("available_at").drop_duplicates("date", keep="last")
    return d.set_index("date")["value"].sort_index()


def run_length(cond):
    """끝에서부터 조건이 연속으로 참인 일수."""
    n = 0
    for v in cond.iloc[::-1]:
        if bool(v):
            n += 1
        else:
            break
    return n


def trend_state(price, k=K_TREND):
    """종가가 200일선 아래로 k거래일 연속이면 '이탈', 위로 k거래일 연속이면 '복귀'(보유). 처음에는 보유로 시작한다."""
    sma = price.rolling(200).mean()
    valid = sma.dropna().index
    if len(valid) == 0:
        return {"state": "자료 부족", "since": None, "pct_vs_200d": None, "run_below": 0, "run_above": 0}
    state, below, above, since = 1, 0, 0, valid[0]
    for d in valid:
        p, m = price.loc[d], sma.loc[d]
        below, above = (below + 1, 0) if p < m else (0, above + 1) if p > m else (0, 0)
        if state == 1 and below >= k:
            state, since = 0, d
        elif state == 0 and above >= k:
            state, since = 1, d
    last = valid[-1]
    return {"state": "보유" if state else "이탈", "since": str(since.date()), "asof": str(last.date()),
            "pct_vs_200d": float(price.loc[last] / sma.loc[last] - 1), "run_below": below, "run_above": above,
            "dd_52w": float(price.iloc[-1] / price.tail(252).max() - 1)}


def level_hy(diff):
    return RED if diff >= 0.5 else YELLOW if diff >= 0.3 else GREEN


def level_jpy(chg):
    return RED if chg <= -5 else YELLOW if chg <= -3 else GREEN


def level_cpi(mom):
    return RED if mom >= 0.3 else GREEN if mom < 0.2 else YELLOW


def level_sahm(v):
    return RED if v >= 0.5 else YELLOW if v >= 0.3 else GREEN


def _ind(key, name, level, value, detail, asof=None):
    return {"id": key, "name": name, "level": level, "value": value, "detail": detail, "asof": asof}


def indicators():
    out = []
    y10 = _fred("DGS10")
    if len(y10):
        run = run_length(y10 > 5.0)
        out.append(_ind("EW1", "10년물 5% 초과", RED if y10.iloc[-1] > 5.0 else GREEN, float(y10.iloc[-1]),
                        f"{y10.iloc[-1]:.2f}% · 5% 초과 연속 {run}일", str(y10.index[-1].date())))
    y2, ff = _fred("DGS2"), _fred("DFEDTARU")
    if len(y2) and len(ff):
        upper = ff.loc[:y2.index[-1]].iloc[-1]
        spread = float(y2.iloc[-1] - upper)
        out.append(_ind("EW2", "2년물 − 정책금리 상단", RED if spread > 0 else GREEN, spread,
                        f"2년물 {y2.iloc[-1]:.2f}% − 상단 {upper:.2f}% = {spread:+.2f}%p", str(y2.index[-1].date())))
    hy = _fred("BAMLH0A0HYM2")
    if len(hy) >= 126:
        w = hy.tail(126)
        diff = float(hy.iloc[-1] - w.min())
        out.append(_ind("EW3", "하이일드, 6개월 최저 대비", level_hy(diff), diff,
                        f"{hy.iloc[-1]:.2f}% · 6개월 최저 {w.min():.2f}%({w.idxmin().date()}) 대비 {diff:+.2f}%p", str(hy.index[-1].date())))
    jpy = _px("JPY=X")
    if len(jpy) > 22:
        chg = float((jpy.iloc[-1] / jpy.iloc[-22] - 1) * 100)
        out.append(_ind("EW4", "USD/JPY 1개월(엔 절상은 하락)", level_jpy(chg), chg, f"{jpy.iloc[-1]:.2f} · 21거래일 {chg:+.1f}%", str(jpy.index[-1].date())))
    un = _fredv("UNRATE")
    if len(un) >= 13:
        last, prev, low = float(un.iloc[-1]), float(un.iloc[-2]), float(un.tail(12).min())
        lvl = RED if last - low >= 0.3 else YELLOW if last > prev else GREEN
        out.append(_ind("EW5", "실업률", lvl, last, f"{last:.1f}% (직전 {prev:.1f}%, 12개월 최저 {low:.1f}% 대비 {last - low:+.1f}%p)", str(un.index[-1].date())))
    sahm = _fred("SAHMREALTIME")
    if len(sahm):
        out.append(_ind("EW6", "삼의 법칙", level_sahm(float(sahm.iloc[-1])), float(sahm.iloc[-1]), f"{sahm.iloc[-1]:.2f} (0.3 노랑, 0.5 빨강)", str(sahm.index[-1].date())))
    above = [(t, bool(_px(t).iloc[-1] > _px(t).rolling(200).mean().iloc[-1])) for t in BREADTH_ETFS if len(_px(t)) > 200]
    if above:
        share = sum(a for _, a in above) / len(above)
        out.append(_ind("EW7", "호흡 근사(ETF 200일선 위 비율)", RED if share < 0.6 else GREEN, share,
                        f"{sum(a for _, a in above)}/{len(above)}개 · 아래: " + ", ".join(t for t, a in above if not a)))
    kis = PARQUET / "kis_investor_kospi.parquet"
    if kis.exists():
        f = pd.read_parquet(kis)["frgn_ntby_tr_pbmn"].dropna()
        sell = run_length(f < 0)
        buy = run_length(f > 0)
        streak = f"연속 순매도 {sell}일" if sell else f"연속 순매수 {buy}일"
        out.append(_ind("EW8", "외국인 코스피 순매매", YELLOW if sell >= 5 else GREEN, float(f.tail(20).sum()),
                        f"{streak} · 최근 20일 합 {f.tail(20).sum() / 1e6:+.1f}조 원(단위 백만 원 가정)", str(f.index[-1].date())))
    cpi = _fredv("CPILFESL").asfreq("MS")
    mom = ((cpi / cpi.shift(1) - 1) * 100).dropna()
    if len(mom):
        out.append(_ind("EW9", "근원 CPI 전월비", level_cpi(float(mom.iloc[-1])), float(mom.iloc[-1]),
                        f"{mom.iloc[-1]:+.2f}% ({mom.index[-1]:%Y-%m})", str(mom.index[-1].date())))
    out.append(_ind("EW10", "빅테크 다음 해 CAPEX 가이던스 증가율", NA, None, "수기 항목(미입력). 20% 미만이면 빨강"))
    return out


def aux():
    spx, vix, nf = _px("^GSPC"), _px("^VIX"), _fred("NFCI")
    out = {}
    if len(vix):
        out["vix"] = {"value": float(vix.iloc[-1]), "run_ge25": run_length(vix >= 25), "run_ge30": run_length(vix >= 30), "asof": str(vix.index[-1].date())}
    if len(spx) > 200:
        s50, s200 = spx.rolling(50).mean(), spx.rolling(200).mean()
        out["spx"] = {"close": float(spx.iloc[-1]), "pct_vs_50d": float(spx.iloc[-1] / s50.iloc[-1] - 1), "pct_vs_200d": float(spx.iloc[-1] / s200.iloc[-1] - 1),
                      "death_cross": bool(s50.iloc[-1] < s200.iloc[-1]), "run_below_200d": run_length(spx < s200), "asof": str(spx.index[-1].date())}
    if len(nf):
        out["nfci"] = {"value": float(nf.iloc[-1]), "asof": str(nf.index[-1].date())}
    return out


def trend_board():
    rows = []
    for name, ticker, owned in TREND_ASSETS:
        p = _px(ticker)
        row = {"name": name, "ticker": ticker, "owned": owned}
        row.update(trend_state(p) if len(p) else {"state": "자료 부족", "since": None, "pct_vs_200d": None})
        rows.append(row)
    return rows


def stage(trend, red, aux_, recent=20):
    """시장 단계(docs/AI_PROFITABILITY.md 2026-11-24 사전 등록). 위에서부터 처음 맞는 조건."""
    by = {t["ticker"]: t for t in trend}
    spx = by.get("^GSPC", {})
    if spx.get("state") not in ("보유", "이탈"):
        return {"code": None, "name": "판정 불가", "why": "S&P 500 자료 부족"}
    vix = aux_.get("vix", {})
    if spx["state"] == "이탈":
        if vix.get("run_ge30", 0) >= 3 or red >= 5:
            return {"code": 4, "name": "위기", "why": f"S&P 이탈 + (VIX 30 이상 연속 {vix.get('run_ge30', 0)}일, 빨강 {red}개)"}
        return {"code": 3, "name": "경계", "why": f"S&P 500 이탈({spx.get('since')}부터)"}
    since = pd.Timestamp(spx["since"])
    days = len(_px("^GSPC").loc[since:]) - 1
    if days <= recent and (by["^GSPC"].get("since") != _first_valid(_px("^GSPC"))):
        return {"code": 0, "name": "회복 관찰", "why": f"S&P 복귀 후 {days}거래일(20거래일까지)"}
    if red >= 2 or any(by.get(k, {}).get("state") == "이탈" for k in ("QQQ", "SOXX")):
        return {"code": 2, "name": "주의", "why": f"빨강 {red}개, QQQ·SOXX 이탈: " + ", ".join(k for k in ("QQQ", "SOXX") if by.get(k, {}).get("state") == "이탈")}
    return {"code": 1, "name": "정상", "why": "해당 경고 없음"}


def _first_valid(price):
    v = price.rolling(200).mean().dropna().index
    return str(v[0].date()) if len(v) else None


def compute():
    ind = indicators()
    tb, ax = trend_board(), aux()
    red = sum(i["level"] == RED for i in ind)
    return {"indicators": ind, "aux": ax, "trend": tb, "stage": stage(tb, red, ax),
            "red": red, "yellow": sum(i["level"] == YELLOW for i in ind), "green": sum(i["level"] == GREEN for i in ind)}


def summary(ew):
    """장부용 요약: 지표별 신호등과 자산별 추세 상태."""
    return {"levels": {i["id"]: i["level"] for i in ew["indicators"]}, "trend": {t["ticker"]: t["state"] for t in ew["trend"]},
            "stage": ew["stage"]["name"]}
