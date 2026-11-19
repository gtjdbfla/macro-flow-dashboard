"""보유 업종 점검 — 메모리·광통신·전력. 표시만 하며 판정·등급을 만들지 않는다(docs/AI_PROFITABILITY.md, 2026-11-19 사전 등록)."""
import pandas as pd

from src.ai.end_demand import basket
from src.data import PARQUET
from src.data.sec import MEMORY, OPTICAL, OWNED, POWER
from src.data.yahoo import _fname

GROUPS = (("메모리", MEMORY), ("광통신", OPTICAL), ("전력 인프라", POWER), ("보유 개별 종목", OWNED))
NAMES = {"MU": "Micron", "LITE": "Lumentum", "COHR": "Coherent", "CIEN": "Ciena", "GLW": "Corning", "FN": "Fabrinet",
         "CRDO": "Credo", "ANET": "Arista", "VRT": "Vertiv", "GEV": "GE Vernova", "ETN": "Eaton", "PWR": "Quanta Services",
         "HUBB": "Hubbell", "NVDA": "NVIDIA", "000660.KS": "SK하이닉스", "0173Y0.KS": "KODEX 미국AI광통신네트워크",
         "AIPO": "AIPO(Defiance AI·전력 인프라 ETF)", "BE": "Bloom Energy", "FRMI": "Fermi"}
# 사용자가 직접 보유한다고 밝힌 종목(2026-11-19). 나머지는 업종 대표 비교용이다.
OWNED_TICKERS = ("NVDA", "000660.KS", "0173Y0.KS", "AIPO", "BE", "FRMI")
# 10-Q 원문에서 확인한 개별 종목 사실(억 달러). 새 공시가 나오면 일일 점검 루틴이 갱신한다.
FERMI_10Q = "https://www.sec.gov/Archives/edgar/data/2071778/000207177826000058/frmi-20260930.htm"
FERMI_FACTS = (
    ("2026-09-30", "매출", "0 (아직 매출 없음)"),
    ("2026-09-30", "현금·제한 현금", "2.86억 + 0.29억 달러"),
    ("2026-09-30", "차입금(순)", "9.58억 달러 (2025년 말 1.10억)"),
    ("2026-09-30", "계속기업 의문", "감사 전 분기보고서에 '1년 내 의무를 이행하기에 자원이 부족해 계속기업 의문(substantial doubt)'이라고 기재"),
    ("2026-12-31", "Texas Tech 토지 임대 착공 통지(NTP) 기한", "기한 안에 받지 못하면 연장을 받거나 임대가 해지될 수 있음"),
    ("미정", "확정 임차인 계약", "본문 문구상 건설 착수는 확정 임대 체결과 NTP 이후로 서술됨(체결 사실을 찾지 못함)"),
)


def _rpo_yoy(sec, ticker, as_of):
    d = sec[(sec.ticker == ticker) & (sec.metric == "rpo") & (sec.filed <= as_of)].sort_values("filed").drop_duplicates("end", keep="first")
    d = d.set_index("end").value.sort_index()
    if d.empty:
        return None
    last = d.index[-1]
    near = d.index[abs((d.index - (last - pd.DateOffset(years=1))).days) <= 10]
    return float(d.iloc[-1] / d[near[-1]] - 1) if len(near) else None


def fundamentals(sec, as_of=None):
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    out = []
    for group, tickers in GROUPS:
        t, _ = basket(sec, tickers, NAMES, as_of)
        t.insert(0, "group", group)
        t["rpo_yoy"] = [_rpo_yoy(sec, tk, as_of) for tk in t.ticker]
        out.append(t)
    return pd.concat(out, ignore_index=True)


def prices(as_of=None):
    """종가(수정주가 우선)의 52주 고점 대비 낙폭과 1·3개월 수익률."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    rows = []
    for ticker, name in NAMES.items():
        f = PARQUET / _fname(ticker)
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        s = (d["adjclose"] if "adjclose" in d else d["close"]).dropna()
        s = s[s.index <= as_of]
        if len(s) < 70:
            continue
        last = s.iloc[-1]
        ret = lambda days: float(last / s.iloc[-1 - days] - 1)  # noqa: E731  거래일 기준
        rows.append({"ticker": ticker, "name": name, "owned": ticker in OWNED_TICKERS, "last_date": s.index[-1], "dd_52w": float(last / s.tail(252).max() - 1),
                     "ret_1m": ret(21), "ret_3m": ret(63)})
    return pd.DataFrame(rows)
