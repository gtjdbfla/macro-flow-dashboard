"""AI 인프라 기업의 신규 채권 발행 조건 — SEC FWP(최종 조건표)·8-K Item 2.03 파싱.

신용 스트레스의 무료 직접 지표: '새로 발행할 때 시장이 요구한 가산금리(스프레드)'. 통화가 달라도(USD·GBP·EUR) 벤치마크 대비 스프레드(bp)는
비교 가능하다. 파싱은 정규식이라 형식이 다른 문서는 일부 필드가 비고(NaN), 그런 값은 만들지 않는다(추정 금지).

한계: CDS·유통시장 수익률이 아니라 '발행 시점'만 보이므로 발행이 없는 기간은 빈칸이다. 회사 신용등급이 달라(AA 아마존 vs HY 코어위브)
수준 비교보다 같은 회사의 시계열 변화를 본다."""
import re
import time

import pandas as pd
import requests

from . import PARQUET

UA = {"User-Agent": "macro-flow-dashboard research gtjdbfla@gmail.com"}
CIKS = {"ORCL": "0001341439", "META": "0001326801", "GOOGL": "0001652044", "AMZN": "0001018724", "MSFT": "0000789019",
        "NVDA": "0001045810", "CRWV": "0001769628"}
BENCH = r"(?:Benchmark )?(?:Treasury|Treasuries|Gilt|Bund|Mid-?Swaps?|Swap|Benchmark)"


def _get(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def _clean(t):
    import html
    t = re.sub(r"<(script|style).*?</\1>", "", t, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def list_filings(ticker: str) -> list:
    r = _get(f"https://data.sec.gov/submissions/CIK{CIKS[ticker]}.json")
    f = r.json()["filings"]["recent"]
    out = []
    for i in range(len(f["form"])):
        form, items = f["form"][i], f["items"][i]
        if form == "FWP" or (form == "8-K" and "2.03" in items and ticker == "CRWV"):
            out.append((f["filingDate"][i], f["accessionNumber"][i], f["primaryDocument"][i], form))
    return out


def parse_8k_notes(t: str, filed: str) -> list:
    """8-K Item 1.01/2.03: 'x.xxx% Senior Notes due YYYY'와 그 앞 금액. 대출(SOFR+마진) 설비는 포함하지 않는다."""
    i = t.find("Item 1.01")
    seg = t[i:] if i >= 0 else t
    tr, seen = [], set()
    for c, y in re.findall(r"(\d{1,2}\.\d{3})\s?%\s+(?:Senior |Senior Secured )?Notes due (\d{4})", seg):
        if (c, y) not in seen:
            seen.add((c, y))
            tr.append((float(c), int(y)))
    if not tr:
        return []
    amts = re.findall(r"([$€£])\s?([\d,]+)(?:\s?(million|billion))?\s+(?:in\s+)?aggregate principal amount", seg)
    cur_map = {"$": "USD", "€": "EUR", "£": "GBP"}
    rows = []
    for k, (c, y) in enumerate(tr):
        a = float("nan")
        cur = "EUR" if re.search(r"euro-denominated %s" % re.escape("%.3f%%" % c), seg) else "USD"
        if len(amts) >= len(tr):
            sym, num, unit = amts[k]
            v = float(num.replace(",", ""))
            a = v * (1e6 if unit == "million" else 1e9 if unit == "billion" else (1e6 if v < 1e5 else 1))
            if a < 5e7:  # 공시 오타($1,750,000 등)는 만들지 않는다
                a = float("nan")
            cur = cur_map.get(sym, cur)
        rows.append({"kind": "debt", "currency": cur, "coupon": c, "maturity_year": y, "tenor": y - pd.Timestamp(filed).year,
                     "spread_bp": float("nan"), "ytm": c, "amount": a})
    return rows


def parse_term_sheet(t: str, filed: str) -> list:
    head = t[:2500]
    # 통화: 문서 전체에서 통화 기호/명칭이 가장 많이 나오는 것 (머리말의 'Euronext' 같은 우연한 단어 방지)
    counts = {"GBP": len(re.findall(r"£|Sterling", t)), "EUR": len(re.findall(r"€|Euro|EUR", t)),
              "CHF": len(re.findall(r"CHF|Swiss franc", t)), "JPY": len(re.findall(r"JPY|Japanese Yen", t)),
              "USD": len(re.findall(r"US\$|U\.S\. dollar|\$\s?\d", t)) // 3}
    cur = max(counts, key=counts.get)
    if re.search(r"Preferred Stock|Depositary Shares|Convertible Notes|Common Stock Offering", head[:900]) and not re.search(r"Notes due", head):
        return [{"kind": "equity/preferred", "currency": cur}]
    tr = re.findall(r"(\d{1,2}\.\d{2,3})\s?%\s+(?:Fixed[- ]Rate )?(?:Senior |Sustainability |Green )?(?:Notes?|Debentures|Bonds)\s+due\s+(?:[A-Za-z]+\s+\d{1,2},\s+)?(\d{4})", head)
    seen, tranches = set(), []
    for c, y in tr:
        if (c, y) not in seen:
            seen.add((c, y))
            tranches.append((float(c), int(y)))
    if not tranches:
        return []
    n = len(tranches)
    sp = re.search(r"Spread to %s[^:]{0,40}:\s*(.{0,420}?)(?:Benchmark|Interest Payment|Price to|Optional|Yield|CUSIP|Day Count|$)" % BENCH, t)
    spreads = []
    if sp:
        spreads = [float(x) for x in re.findall(r"\+\s*(\d+(?:\.\d+)?)\s*(?:bps|basis points)", sp.group(1))]
        if not spreads:
            spreads = [float(x) for x in re.findall(r"T\s*\+\s*(\d+(?:\.\d+)?)", sp.group(1))]
    if not spreads:
        spreads = [float(x) for x in re.findall(r"T\s*\+\s*(\d+(?:\.\d+)?)\s*(?:bps|basis)", t)][:n] or []
    ytm = []
    m = re.search(r"Yield to Maturity[^:]{0,30}:\s*(.{0,300}?)(?:Spread|Benchmark|Price|Interest)", t)
    if m:
        ytm = [float(x) for x in re.findall(r"(\d{1,2}\.\d{2,3})\s?%", m.group(1))]
    if not ytm:
        ytm = [float(x) for x in re.findall(r"(?:Re-?offer Yield|Yield to Maturity)\s*:?\s*(\d{1,2}\.\d{2,3})\s?%", t)]
    pr = re.search(r"(?:Aggregate )?Principal Amount[^:]{0,40}:\s*(.{0,360}?)(?:Maturity|Coupon|Ratings|Issue Price|Price to|Trade Date)", t)
    amounts = []
    if pr:
        amounts = [float(x.replace(",", "")) for x in re.findall(r"[$€£]\s?([\d,]{9,})", pr.group(1))]
        if not amounts:
            amounts = [float(x) * 1e6 for x in re.findall(r"[$€£]\s?([\d,.]+)\s*million", pr.group(1), re.I)]
    rows = []
    for i, (c, y) in enumerate(tranches):
        pick = lambda lst: lst[i] if len(lst) == n else (lst[0] if len(lst) == 1 and n == 1 else float("nan"))  # noqa: E731
        rows.append({"kind": "debt", "currency": cur, "coupon": c, "maturity_year": y, "tenor": y - pd.Timestamp(filed).year,
                     "spread_bp": pick(spreads), "ytm": pick(ytm), "amount": pick(amounts)})
    return rows


def collect(tickers=None) -> dict:
    rows, res = [], {}
    for tk in tickers or CIKS:
        n_ok = n_bad = 0
        try:
            fl = list_filings(tk)
        except Exception as e:
            res[tk] = {"error": str(e)[:150]}
            continue
        for filed, acc, doc, form in fl:
            r = _get(f"https://www.sec.gov/Archives/edgar/data/{int(CIKS[tk])}/{acc.replace('-', '')}/{doc}")
            time.sleep(0.15)
            if r is None:
                n_bad += 1
                continue
            txt = _clean(r.text)
            parsed = parse_8k_notes(txt, filed) if form == "8-K" else parse_term_sheet(txt, filed)
            if not parsed:
                n_bad += 1
                continue
            n_ok += 1
            for p in parsed:
                rows.append({"ticker": tk, "filed": pd.Timestamp(filed), "form": form, "accession": acc, **p})
        res[tk] = {"parsed": n_ok, "no_terms": n_bad}
    if rows:
        df = pd.DataFrame(rows)
        # CoreWeave 8-K는 이전 발행분을 다시 언급하므로 (쿠폰, 만기) 최초 등장만 신규 발행으로 본다(추가 발행분은 반영 안 됨)
        is_crwv = df.ticker == "CRWV"
        df = pd.concat([df[~is_crwv], df[is_crwv].sort_values("filed").drop_duplicates(["coupon", "maturity_year", "currency"], keep="first")])
        old_f = PARQUET / "debt_issuance.parquet"
        df.sort_values(["ticker", "filed"]).to_parquet(old_f)
    return res
