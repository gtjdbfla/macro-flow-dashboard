"""GPU 임대가격 분석 — 공급과잉 신호를 '신형 가격 하락'과 '스팟 할인 확대'로 읽는다. 데이터: src/data/gpurent.py (게시가격, 체결가 아님)."""
import pandas as pd

from src.data import PARQUET

GPUS = ["H100", "H200", "B200", "B300"]
NEW = ["B200", "B300"]
DROP = -0.10   # F15: 신형 7일 평균이 28일 전 7일 평균보다 10% 이상 하락 (사전 고정)
MIN_DAYS = 35  # 7일 평균 + 28일 전 비교에 필요한 최소 관측 일수. 미만이면 판정 보류


def series() -> pd.DataFrame:
    """전 세계·전체 공급업체 기준 (day × product/tenor) 가격."""
    d = pd.read_parquet(PARQUET / "gpu_rental.parquet")
    g = d[(d.region == "GLB") & (d.tier == "ALL") & d["product"].isin(GPUS)]
    return g.pivot_table(index="day", columns=["product", "tenor"], values="usd_per_gpu_hour")


def latest(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GPUS:
        od, sp = p[(g, "OD")].dropna(), p[(g, "SPOT")].dropna()
        yr = p[(g, "1Y")].dropna() if (g, "1Y") in p else pd.Series(dtype=float)
        rows.append({"gpu": g, "od": od.iloc[-1], "spot": sp.iloc[-1] if len(sp) else float("nan"),
                     "spot_ratio": sp.iloc[-1] / od.iloc[-1] if len(sp) else float("nan"),
                     "od_7d": od.iloc[-1] / od.iloc[-8] - 1 if len(od) > 7 else float("nan"),
                     "yr1": yr.iloc[-1] if len(yr) else float("nan"), "days": len(od)})
    return pd.DataFrame(rows)


def flags(p: pd.DataFrame) -> list:
    n = p[("B200", "OD")].dropna().shape[0]
    if n < MIN_DAYS:
        return [{"id": "F15", "title": "신형 GPU 임대가격(B200·B300)", "on": False, "level": "보류",
                 "text": "관측 %d일뿐이라 판정하지 않음(7일 평균을 28일 전과 비교하려면 %d일 필요). 공개 CSV가 2026-10-26부터라 수집 누적 후 자동으로 판정된다. "
                         "참고: 최근 7일 B200 %+.1f%%, B300 %+.1f%%." % (
                             n, MIN_DAYS, *[(p[(g, "OD")].dropna().iloc[-1] / p[(g, "OD")].dropna().iloc[-8] - 1) * 100 for g in NEW])}]
    ch = {}
    for g in NEW:
        s = p[(g, "OD")].dropna()
        ch[g] = s.tail(7).mean() / s.iloc[-35:-28].mean() - 1
    k = sum(v <= DROP for v in ch.values())
    lvl = "경고" if k == len(NEW) else ("주의" if k else "정상")
    return [{"id": "F15", "title": "신형 GPU 임대가격(B200·B300)", "on": k > 0, "level": lvl,
             "text": "온디맨드 7일 평균의 4주 전 대비: " + ", ".join("%s %+.1f%%" % (g, v * 100) for g, v in ch.items()) + " — %d개가 %d%% 이하 하락." % (k, abs(DROP) * 100)}]
