"""SEC XBRL(companyfacts) -> AI 인프라 기업 분기 재무. 공시일(filed) 포함 = point-in-time.

현금흐름·손익 항목은 10-Q가 누적(YTD)으로 공시되는 경우가 많아, 같은 회계연도 시작일의 누적 구간끼리 차분해 분기 값을 만든다
(Q4 = 연간 − 9개월 누적). 파생 분기값의 공시일은 구성 값 중 늦은 쪽.
같은 (구간) 값이 여러 번 공시(정정·재공시)되면 **최초 공시값**을 쓴다(당시 시장이 본 값; 후행 재작성 제외).
태그는 회사·시기별로 바뀌므로 우선순위 목록에서 해당 분기 값이 있는 첫 태그를 쓴다."""
import time

import pandas as pd
import requests

from . import PARQUET

UA = {"User-Agent": "macro-flow-dashboard research gtjdbfla@gmail.com"}
CIKS = {"MSFT": "0000789019", "GOOGL": "0001652044", "AMZN": "0001018724", "META": "0001326801",
        "ORCL": "0001341439", "CRWV": "0001769628", "NVDA": "0001045810"}
# F18 소프트웨어 AI 매출화 바스켓(2026-11-18 사전 고정, docs/AI_PROFITABILITY.md). 상장폐지·합병이 아니면 바꾸지 않는다.
SOFTWARE = {"CRM": "0001108524", "NOW": "0001373715", "ADBE": "0000796343", "INTU": "0000896878", "PLTR": "0001321655",
            "WDAY": "0001327811", "SNOW": "0001640147", "DDOG": "0001561550", "MDB": "0001441816", "HUBS": "0001404655"}
# 보유 업종 점검 바스켓(2026-11-19 사전 고정, docs/AI_PROFITABILITY.md): 메모리·광통신·전력.
MEMORY = {"MU": "0000723125"}
OPTICAL = {"LITE": "0001633978", "COHR": "0000820318", "CIEN": "0000936395", "GLW": "0000024741", "FN": "0001408710",
           "CRDO": "0001807794", "ANET": "0001596532"}
POWER = {"VRT": "0001674101", "GEV": "0001996810", "ETN": "0001551182", "PWR": "0001050915", "HUBB": "0000048898"}
CIKS = {**CIKS, **SOFTWARE, **MEMORY, **OPTICAL, **POWER}
FLOW_TAGS = {
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "cost_rev": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
           "DepreciationAndAmortization", "Depreciation"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "interest": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestExpenseDebtExcludingAmortization"],
}
INSTANT_TAGS = {
    "rpo": ["RevenueRemainingPerformanceObligation"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    # 신용 안전성(2026-10-15): 회사마다 태그가 달라 우선순위 목록에서 해당 분기 값이 있는 첫 태그를 쓴다
    "debt_nc": ["LongTermDebtNoncurrent", "LongTermNotesAndLoans", "LongTermNotesPayable", "LongTermDebt"],
    "debt_c": ["DebtCurrent", "LongTermDebtCurrent", "NotesPayableCurrent", "ShortTermBorrowings"],
    "debt_carry": ["DebtInstrumentCarryingAmount"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "st_inv": ["MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "ShortTermInvestments"],
    "cash_sti": ["CashCashEquivalentsAndShortTermInvestments", "CashCashEquivalentsAndMarketableSecurities"],
    "op_lease": ["OperatingLeaseLiability"],
    "fin_lease": ["FinanceLeaseLiability"],
    # 수요의 질(F16, 2026-11-18): NVIDIA DSO·재고
    "ar": ["AccountsReceivableNetCurrent"],
    "inventory": ["InventoryNet"],
    "debt_due_12m": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"],
    "debt_y2": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo"],
    "debt_y3": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree"],
    "debt_y4": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour"],
    "debt_y5": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive"],
    "debt_after5": ["LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive"],
}


def _facts(cf: dict, tag: str, unit: str = "USD"):
    f = cf["facts"].get("us-gaap", {}).get(tag)
    if not f or unit not in f.get("units", {}):
        return []
    out = []
    for x in f["units"][unit]:
        if not str(x.get("form", "")).startswith("10-"):
            continue
        out.append({"start": pd.to_datetime(x["start"]) if x.get("start") else pd.NaT, "end": pd.to_datetime(x["end"]),
                    "val": float(x["val"]), "filed": pd.to_datetime(x["filed"])})
    return out


def _first_filed(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("filed").drop_duplicates(["start", "end"], keep="first")


def _quarterize(rows) -> dict:
    """{end: (value, filed)} 분기 값."""
    df = _first_filed(rows)
    if df.empty:
        return {}
    df = df.dropna(subset=["start"])
    df["days"] = (df["end"] - df["start"]).dt.days
    out = {}
    for _, r in df[(df.days >= 80) & (df.days <= 100)].iterrows():
        out[r["end"]] = (r["val"], r["filed"])
    ytd = df[df.days > 100]
    for _, r in ytd.iterrows():
        if r["end"] in out:
            continue
        prev = df[(df.start == r["start"]) & ((r["end"] - df.end).dt.days.between(80, 100))]
        if len(prev):
            p = prev.iloc[0]
            out[r["end"]] = (r["val"] - p["val"], max(r["filed"], p["filed"]))
    return out


def fetch_company(ticker: str) -> pd.DataFrame:
    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIKS[ticker]}.json", headers=UA, timeout=90)
    r.raise_for_status()
    cf = r.json()
    rows = []
    for metric, tags in FLOW_TAGS.items():
        got = {}
        for tag in tags:
            for end, (v, filed) in _quarterize(_facts(cf, tag)).items():
                if end not in got:
                    got[end] = (v, filed, tag)
        rows += [{"ticker": ticker, "metric": metric, "end": e, "value": v, "filed": f, "tag": t}
                 for e, (v, f, t) in got.items()]
    for metric, tags in INSTANT_TAGS.items():
        got = {}
        for tag in tags:
            df = _first_filed([x for x in _facts(cf, tag) if pd.isna(x["start"])])
            for _, x in df.iterrows():
                if x["end"] not in got:
                    got[x["end"]] = (x["val"], x["filed"], tag)
        rows += [{"ticker": ticker, "metric": metric, "end": e, "value": v, "filed": f, "tag": t}
                 for e, (v, f, t) in got.items()]
    sh = _first_filed(_facts(cf, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"))
    if not sh.empty:
        sh = sh.dropna(subset=["start"])
        sh = sh[((sh["end"] - sh["start"]).dt.days >= 80) & ((sh["end"] - sh["start"]).dt.days <= 100)]
        rows += [{"ticker": ticker, "metric": "shares_dil", "end": r["end"], "value": r["val"], "filed": r["filed"],
                  "tag": "WeightedAverageNumberOfDilutedSharesOutstanding"} for _, r in sh.iterrows()]
    return pd.DataFrame(rows)


def collect(tickers=None) -> dict:
    out, frames = {}, []
    for t in (tickers or CIKS):
        try:
            d = fetch_company(t)
            frames.append(d)
            out[t] = {"start": str(d["end"].min().date()), "end": str(d["end"].max().date()), "n": len(d),
                      "latest_filed": str(d["filed"].max().date())}
        except Exception as e:
            out[t] = {"error": str(e)[:200]}
        time.sleep(0.3)
    if frames:
        pd.concat(frames).sort_values(["ticker", "metric", "end"]).to_parquet(PARQUET / "sec_quarterly.parquet")
    return out
