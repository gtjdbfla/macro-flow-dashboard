"""클라우드 부문 실적(Azure/AWS/Google Cloud) — SEC 8-K Item 2.02 실적 발표문(Exhibit 99.1) 파싱.

부문 매출·영업이익은 XBRL companyfacts에 없어(차원 데이터) 발표문 표를 정규식으로 읽는다. 각 발표문은 '당 분기'와 '전년 동기'를
함께 싣기 때문에 YoY를 그 발표문 안에서 계산한다(부문 재편성 영향 최소화). 파싱에 실패한 분기는 값을 만들지 않고 N/A로 둔다.
또 경영진이 공개한 AI 관련 연환산(run-rate) 문장은 정량 데이터와 섞지 않고 ai_disclosures에 원문 그대로 따로 저장한다.

quarter_end는 발표일 직전 달력 분기말(세 회사 모두 달력 분기 기준 보고, MSFT 회계연도 분기도 달력 분기말과 일치)."""
import html
import re
import time

import pandas as pd
import requests

from . import PARQUET

UA = {"User-Agent": "macro-flow-dashboard research gtjdbfla@gmail.com"}
CIKS = {"MSFT": "0000789019", "GOOGL": "0001652044", "AMZN": "0001018724"}


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


def earnings_filings(ticker: str) -> list:
    """[(filed, accession)] 최신순. recent 목록에 있는 Item 2.02 8-K."""
    r = _get(f"https://data.sec.gov/submissions/CIK{CIKS[ticker]}.json")
    f = r.json()["filings"]["recent"]
    return [(f["filingDate"][i], f["accessionNumber"][i]) for i in range(len(f["form"]))
            if f["form"][i] == "8-K" and "2.02" in f["items"][i]]


def exhibit_text(ticker: str, accession: str):
    cik = str(int(CIKS[ticker]))
    r = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/index.json")
    if r is None:
        return None
    names = [x["name"] for x in r.json()["directory"]["item"]]
    cand = [n for n in names if re.search(r"ex-?99[_-]?1|exhibit99|ex99", n, re.I) and n.lower().endswith((".htm", ".html"))]
    if not cand:
        return None
    d = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{cand[0]}")
    if d is None:
        return None
    t = re.sub(r"<(script|style).*?</\1>", "", d.text, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t))


def quarter_end(filed: str) -> pd.Timestamp:
    d = pd.Timestamp(filed)
    q = (d - pd.Timedelta(days=1)).to_period("Q")
    prev = q - 1 if (d - q.end_time.normalize()).days <= 0 else q
    # 발표일이 속한 분기의 직전 분기말
    cur = d.to_period("Q")
    return (cur - 1).end_time.normalize()


def _num(s):
    s = s.strip()
    neg = s.startswith("(")
    v = float(s.strip("()$ ").replace(",", ""))
    return -v if neg else v


N = r"\(?[\d,]+\)?"


def parse_msft(t):
    o = {}
    # 1) 세그먼트 표(백만달러, 당기·전년 순): 가장 정확
    m = re.search(r"Intelligent Cloud \$?\s*(%s)\s+\$?\s*(%s)\s+(?:\$?\s*%s\s+\$?\s*%s\s+)?More Personal Computing" % (N, N, N, N), t)
    if m:
        o.update(cloud_rev=_num(m.group(1)), cloud_rev_prior=_num(m.group(2)))
    if "cloud_rev" not in o:
        m = re.search(r"Intelligent Cloud \$?\s*(%s)\s+\$?\s*(%s)\s+\(?(-?\d+)\)?\s?%%" % (N, N), t)
        if m:
            o.update(cloud_rev=_num(m.group(1)), cloud_rev_prior=_num(m.group(2)))
    if "cloud_rev" not in o:
        m = re.search(r"(?:Revenue in )?Intelligent Cloud (?:revenue )?was \$([\d.]+) billion and increased (\d+)%", t)
        if m:
            cur = float(m.group(1)) * 1000
            o.update(cloud_rev=cur, cloud_rev_prior=cur / (1 + int(m.group(2)) / 100))
    # 영업이익: 'Operating Income' 구역의 Intelligent Cloud 행 또는 부문 손익 표
    i = t.find("Operating Income Productivity")
    seg = t[i:i + 1500] if i >= 0 else ""
    m = re.search(r"Intelligent Cloud \$?\s*(%s)\s+\$?\s*(%s)" % (N, N), seg)
    if not m:
        m = re.search(r"Intelligent Cloud Revenue \$?\s*%s\s+\$?\s*%s.{0,260}?Operating income \$?\s*(%s)\s+\$?\s*(%s)" % (N, N, N, N), t)
    if m:
        o.update(cloud_opinc=_num(m.group(1)), cloud_opinc_prior=_num(m.group(2)))
    m = re.search(r"Azure(?: and other cloud services)? revenue (?:increased|grew|growth of) (\d+)%", t)
    if m:
        o["azure_yoy"] = int(m.group(1)) / 100
    return o


def parse_googl(t):
    o = {}
    m = re.search(r"Google Cloud\s+\$?\s*(%s)\s+\$?\s*(%s)\s+Other Bets" % (N, N), t)
    if m:
        o.update(cloud_rev_prior=_num(m.group(1)), cloud_rev=_num(m.group(2)))
    m = re.search(r"Operating income \(loss\):.{0,200}?Google Cloud\s+\$?\s*(%s)\s+\$?\s*(%s)" % (N, N), t)
    if m:
        o.update(cloud_opinc_prior=_num(m.group(1)), cloud_opinc=_num(m.group(2)))
    return o


def parse_amzn(t):
    o = {}
    extra = r"(?:\$?\s*%s\s+\$?\s*%s\s+)?" % (N, N)  # 6개월·연간 누적 열이 있을 수도, 없을 수도 있다
    m = re.search(r"AWS Net sales \$?\s*(%s)\s+\$?\s*(%s)\s+%sOperating expenses\s+%s\s+%s\s+%sOperating income \$?\s*(%s)\s+\$?\s*(%s)"
                  % (N, N, extra, N, N, extra, N, N), t)
    if m:
        o.update(cloud_rev_prior=_num(m.group(1)), cloud_rev=_num(m.group(2)),
                 cloud_opinc_prior=_num(m.group(3)), cloud_opinc=_num(m.group(4)))
    return o


PARSERS = {"MSFT": parse_msft, "GOOGL": parse_googl, "AMZN": parse_amzn}
AI_RE = re.compile(r"[^.•]*\b(AI|Copilot|Gemini|chips?|tokens?|Trainium|GPU|TPU)\b[^.•]*(run[- ]rate|annualized|paid seats|tokens)[^.•]*[.•]?", re.I)


def disclosures(t):
    out = []
    for m in AI_RE.finditer(t):
        s = m.group(0).strip(" •")
        if re.search(r"\d", s) and 30 < len(s) < 420:
            out.append(s)
    return out[:6]


def collect(tickers=("MSFT", "GOOGL", "AMZN"), max_filings=28) -> dict:
    rows, disc, res = [], [], {}
    for tk in tickers:
        ok = bad = 0
        for filed, acc in earnings_filings(tk)[:max_filings]:
            t = exhibit_text(tk, acc)
            time.sleep(0.2)
            if not t:
                bad += 1
                continue
            p = PARSERS[tk](t)
            qe = quarter_end(filed)
            if "cloud_rev" in p:
                ok += 1
            else:
                bad += 1
            rows.append({"ticker": tk, "quarter_end": qe, "filed": pd.Timestamp(filed), "accession": acc, **p})
            for s in disclosures(t):
                disc.append({"ticker": tk, "quarter_end": qe, "filed": pd.Timestamp(filed), "text": s})
        res[tk] = {"parsed": ok, "failed": bad}
    if rows:
        df = pd.DataFrame(rows).sort_values(["ticker", "quarter_end"])
        df.to_parquet(PARQUET / "cloud_segments.parquet")
        pd.DataFrame(disc).drop_duplicates(["ticker", "text"]).to_parquet(PARQUET / "ai_disclosures.parquet")
    return res
