"""Korea Transmission 입력. 행 t(미국 마감 시점)에는 한국 t일 정규장(미국 t일 개장 전에 종료)까지 반영된다.
따라서 '미국 t -> 한국 t+1' 전달 효과는 lead/lag 단계에서 한국 시계열을 shift(-1)로 정렬해 본다.
투자자 수급은 누적합을 달력(미국 영업일)에 올린 뒤 차분 -> 미국 휴장일에 낀 한국 수급도 합산된다."""
import numpy as np
import pandas as pd

from src.data import PARQUET
from src.data.universe import KR_SECTOR_STOCKS

from .common import Ctx, feats_series

FLOW_COLS = {"frgn_ntby_tr_pbmn": "foreign", "orgn_ntby_tr_pbmn": "inst", "prsn_ntby_tr_pbmn": "retail"}
KR_SECTORS = {"091160.KS": "Semis", "091180.KS": "Auto", "091170.KS": "Banks", "143860.KS": "Healthcare",
              "117460.KS": "EnergyChem", "305720.KS": "Battery", "466920.KS": "Shipbuilding",
              "463250.KS": "Defense"}


def build(ctx: Ctx) -> pd.DataFrame:
    parts = []
    # 지수·환율
    for sym, name in [("^KS11", "KOSPI"), ("^KQ11", "KOSDAQ"), ("069500.KS", "KODEX200")]:
        c = ctx.close(sym)
        if c is not None:
            parts.append(pd.DataFrame({f"{name}__ret_{h}d": c.pct_change(h, fill_method=None)
                                       for h in (1, 5, 20, 60)}))
    fx = ctx.close("KRW=X", adj=False)
    if fx is not None:
        parts.append(feats_series(fx.dropna(), "USDKRW", "pct", with_rank=True).drop(columns="USDKRW__level")
                     .reindex(ctx.cal))
    # 업종 ETF 상대강도(KOSPI 대비)
    kospi = ctx.close("^KS11")
    for sym, name in KR_SECTORS.items():
        c = ctx.close(sym)
        if c is None or kospi is None:
            continue
        rs = np.log(c) - np.log(kospi)
        parts.append(pd.DataFrame({f"KR_{name}__rs_{h}d": rs.diff(h) for h in (5, 20, 60)}
                                  | {f"KR_{name}__ret_20d": c.pct_change(20, fill_method=None)}))
    # 투자자별 순매수 (KIS, 백만원)
    for mkt in ("kospi", "kosdaq"):
        f = PARQUET / f"kis_investor_{mkt}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if ctx.asof is not None:
            d = d[d.index <= ctx.asof]
        for col, who in FLOW_COLS.items():
            if col not in d:
                continue
            x = d[col].dropna()
            cum = x.cumsum()
            cum = cum.reindex(cum.index.union(ctx.cal)).ffill().reindex(ctx.cal)
            # 구간 안에 한국 거래일이 하나도 없으면(한국만 휴장) 순매수 0이 아니라 '관측 없음'이다.
            sess = pd.Series(1.0, index=x.index).cumsum()
            sess = sess.reindex(sess.index.union(ctx.cal)).ffill().reindex(ctx.cal)
            daily = cum.diff().where(sess.diff() > 0)  # 달력 행 간 합산 순매수
            sd = daily.rolling(252, min_periods=120).std()
            o = {}
            for h in (1, 5, 20, 60):
                s_h = cum.diff(h).where(sess.diff(h) > 0)
                o[f"{mkt.upper()}_{who}__net_{h}d"] = s_h
                o[f"{mkt.upper()}_{who}__net_{h}d_z"] = s_h / (sd * np.sqrt(h))
            o[f"{mkt.upper()}_{who}__pos_frac_20d"] = (daily > 0).rolling(20).mean()
            parts.append(pd.DataFrame(o))
    parts.append(sector_flows(ctx))
    return pd.concat(parts, axis=1).add_prefix("kr__")


def _on_cal(cum: pd.Series, ctx: Ctx) -> pd.Series:
    return cum.reindex(cum.index.union(ctx.cal)).ffill().reindex(ctx.cal)


def sector_flows(ctx: Ctx) -> pd.DataFrame:
    """업종 대표 종목의 외국인·기관 순매수대금을 업종별로 합산 -> 거래대금 대비 순매수 비율(-1~1).
    구성은 현재 대형주 고정(생존편향 있음). 종목별 데이터가 있는 날만 합산한다."""
    out = {}
    for sector, stocks in KR_SECTOR_STOCKS.items():
        frames = []
        for code in stocks:
            f = PARQUET / f"kis_stock_{code}.parquet"
            if f.exists():
                d = pd.read_parquet(f)
                if ctx.asof is not None:
                    d = d[d.index <= ctx.asof]
                d = d[d["acml_tr_pbmn"] > 0]  # 상장 이전 자리표시 행 제외
                d = d.assign(acml_tr_pbmn=d["acml_tr_pbmn"] / 1e6)  # 거래대금은 원, 순매수대금은 백만원 -> 백만원으로 통일
                frames.append(d[["acml_tr_pbmn", "frgn_ntby_tr_pbmn", "orgn_ntby_tr_pbmn"]])
        if not frames:
            continue
        tot = pd.concat(frames).groupby(level=0).sum(min_count=1).sort_index()
        sess = pd.Series(1.0, index=tot.index).cumsum()
        sess = _on_cal(sess, ctx)
        tv = _on_cal(tot["acml_tr_pbmn"].cumsum(), ctx)
        for col, who in (("frgn_ntby_tr_pbmn", "foreign"), ("orgn_ntby_tr_pbmn", "inst")):
            cum = _on_cal(tot[col].cumsum(), ctx)
            for h in (1, 5, 20, 60):
                has = sess.diff(h) > 0
                out[f"{sector}_{who}__ratio_{h}d"] = (cum.diff(h) / tv.diff(h)).where(has)
                if h == 20:
                    out[f"{sector}_{who}__net_20d"] = cum.diff(h).where(has)
    return pd.DataFrame(out, index=ctx.cal)
