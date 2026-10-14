"""AI 투자 금융환경 — 신규 채권 발행 가산금리(SEC FWP)와 네오클라우드 조달 비용.

무료로 구할 수 있는 직접 신용 지표는 '발행 시점의 요구 스프레드'뿐이다(CDS·유통시장 수익률은 유료). 그래서
  * IG 빅테크(GOOGL·AMZN·META·ORCL): 같은 만기 구간(5년 4~6, 10년 9~11, 30년 20+)의 신규 발행 스프레드(벤치마크 대비 bp) 시계열
  * 네오클라우드(CoreWeave): 신규 선순위 채권 쿠폰(HY)
을 보고, 발행이 없는 기간은 빈칸으로 둔다. 스프레드는 벤치마크(UST/Gilt/Bund) 대비라 통화가 달라도 비교 가능."""
import numpy as np
import pandas as pd

from src.data import PARQUET

IG = ["GOOGL", "AMZN", "META", "ORCL"]
BUCKETS = {"5Y": (4, 6), "10Y": (9, 11), "30Y": (20, 41)}
WIDEN_BP = 15  # 사전 고정: 직전 발행 중앙값보다 15bp 이상 넓으면 '확대'


def issuance() -> pd.DataFrame:
    d = pd.read_parquet(PARQUET / "debt_issuance.parquet")
    return d[d.kind == "debt"].copy()


def spread_series(d: pd.DataFrame) -> pd.DataFrame:
    """발행(일자·통화)별·만기구간별 평균 스프레드."""
    x = d[d.ticker.isin(IG)].dropna(subset=["spread_bp", "tenor"]).copy()
    rows = []
    for b, (lo, hi) in BUCKETS.items():
        y = x[(x.tenor >= lo) & (x.tenor <= hi)]
        g = y.groupby(["ticker", "filed", "currency"]).spread_bp.mean().reset_index()
        g["bucket"] = b
        rows.append(g)
    return pd.concat(rows).sort_values(["ticker", "bucket", "filed"]) if rows else pd.DataFrame()


def widening(ss: pd.DataFrame) -> pd.DataFrame:
    """발행사별 최신 발행 vs 직전 발행들의 중앙값(같은 구간·같은 통화군)."""
    out = []
    for (tk, b), g in ss.groupby(["ticker", "bucket"]):
        g = g.sort_values("filed")
        usd = g[g.currency == "USD"]
        gg = usd if len(usd) >= 3 else g
        if len(gg) < 3:
            continue
        last, prior = gg.iloc[-1], gg.iloc[:-1]
        rec = gg[gg.filed >= "2025-01-01"]
        since = float(last.spread_bp - rec.iloc[0].spread_bp) if len(rec) >= 2 else float("nan")
        out.append({"ticker": tk, "bucket": b, "latest": float(last.spread_bp), "latest_date": last.filed,
                    "prior_median": float(prior.spread_bp.median()), "n_prior": len(prior),
                    "chg_bp": float(last.spread_bp - prior.spread_bp.median()), "since2025_bp": since})
    return pd.DataFrame(out)


def flags(d: pd.DataFrame, prices: dict) -> list:
    out = []
    ss = spread_series(d)
    w = widening(ss) if len(ss) else pd.DataFrame()
    if len(w):
        # 발행사당 구간 하나로 판단(중복 계산 방지): 관측 3개 이상인 구간 중 10년 > 5년 > 30년 순 우선
        w = w.assign(_o=w.bucket.map({"10Y": 0, "5Y": 1, "30Y": 2})).sort_values(["ticker", "_o"]).drop_duplicates("ticker")
        n_wide = int((w.chg_bp >= WIDEN_BP).sum())
        lvl = "주의" if n_wide >= 2 else ("중립" if n_wide == 1 else "정상")
        txt = ", ".join("%s %s +%.0fbp(직전 중앙값 %.0f→%.0f)" % (r.ticker, r.bucket, r.chg_bp, r.prior_median, r.latest) if r.chg_bp >= 0
                        else "%s %s %.0fbp(%.0f→%.0f)" % (r.ticker, r.bucket, r.chg_bp, r.prior_median, r.latest) for r in w.itertuples())
        text = "최신 발행의 가산금리를 직전 발행 중앙값과 비교: " + txt + " — %d곳이 %dbp 이상 확대." % (n_wide, WIDEN_BP)
        ref = ["%s %s %+.0fbp" % (r.ticker, r.bucket, r.since2025_bp) for r in w.itertuples() if r.since2025_bp == r.since2025_bp]
        if ref:
            text += " 참고(2025년 이후 첫 발행 대비): " + ", ".join(ref) + "."
        out.append({"id": "F9", "title": "빅테크 신규 발행 가산금리", "on": n_wide >= 2, "level": lvl, "text": text})
    c = d[d.ticker == "CRWV"].sort_values("filed")
    if len(c):
        usd = c[c.currency == "USD"]
        last = usd.iloc[-1]
        stock = prices.get("CRWV")
        dd = (stock.iloc[-1] / stock.tail(252).max() - 1) if stock is not None and len(stock) > 20 else np.nan
        txt = "CoreWeave 신규 USD 선순위채 쿠폰 %.3f%%(%s, 만기 %d) — 투자적격 빅테크(스프레드 약 0.5~1%%p)와 달리 HY 수준. 최근 발행 %d건" % (
            last.coupon, last.filed.strftime("%Y-%m"), int(last.maturity_year), len(c))
        if dd == dd:
            txt += ", 주가 52주 고점 대비 %+.0f%%" % (dd * 100)
        out.append({"id": "F10", "title": "네오클라우드 조달 비용(HY)", "on": bool(last.coupon >= 9.0), "level": "주의" if last.coupon >= 9.0 else "정상",
                    "text": txt + "."})
    return out
