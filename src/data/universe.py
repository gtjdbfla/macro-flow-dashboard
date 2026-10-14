"""수집 대상 정의. 각 항목의 실제 시작일은 coverage.json에 기록된다(가정하지 않는다)."""

# Yahoo 심볼 -> 설명. 조정종가+거래량.
YAHOO = {
    # 미국 지수·섹터 ETF
    "SPY": "S&P500", "QQQ": "Nasdaq100", "IWM": "Russell2000",
    "XLK": "Technology", "XLC": "Communication", "XLY": "Cons.Disc", "XLP": "Cons.Staples",
    "XLF": "Financials", "XLV": "Healthcare", "XLI": "Industrials", "XLE": "Energy",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate",
    # 산업/테마
    "SOXX": "Semiconductor", "SMH": "Semiconductor(VanEck)", "XSD": "Semi(equal)", "IGV": "Software",
    "CIBR": "Cybersecurity", "KBE": "Banks", "KIE": "Insurance", "IBB": "Biotech", "XPH": "Pharma",
    "IHI": "Med Devices", "ITA": "Aerospace&Defense", "XOP": "Oil&Gas E&P", "OIH": "Energy Services",
    "XBI": "Biotech(XBI)", "XAR": "Aerospace&Defense(XAR)", "KRE": "Regional Banks", "XHB": "Homebuilders",
    "XRT": "Retail", "URA": "Uranium/Nuclear", "COPX": "Copper Miners", "XME": "Metals&Mining", "SLX": "Steel",
    # 매크로 가격
    "^VIX": "VIX", "^MOVE": "MOVE", "DX-Y.NYB": "DXY", "CL=F": "WTI", "GC=F": "Gold",
    "^TNX": "US10Y(yahoo)", "HYG": "HY bond ETF", "LQD": "IG bond ETF", "TLT": "20Y+ Treasury",
    # 한국
    "^KS11": "KOSPI", "^KQ11": "KOSDAQ", "KRW=X": "USDKRW",
    "005930.KS": "Samsung", "000660.KS": "SK hynix",
    # AI 인프라 대표 종목(AI 수익성 모듈의 주가 비교용)
    "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "ORCL": "Oracle", "NVDA": "NVIDIA",
    "CRWV": "CoreWeave", "AVGO": "Broadcom", "TSM": "TSMC ADR", "MU": "Micron", "AMD": "AMD",
    # 한국 업종 ETF(상장일부터만 존재). ^KS200은 이력이 없어 KODEX200을 KOSPI200 대용으로 쓴다.
    "069500.KS": "KODEX200", "091160.KS": "KR Semiconductor", "091180.KS": "KR Auto",
    "091170.KS": "KR Banks", "143860.KS": "KR Healthcare/Biotech", "117460.KS": "KR Energy&Chem",
    "305720.KS": "KR Secondary Battery", "466920.KS": "KR Shipbuilding", "463250.KS": "KR Defense",
}

# FRED 시리즈. 월/분기 지표는 vintage(발표시점)로 받는다.
FRED_DAILY = {
    "DFF": "Fed Funds", "DGS2": "UST 2Y", "DGS10": "UST 10Y", "DFII10": "10Y Real Yield",
    "T10YIE": "10Y Breakeven", "T10Y2Y": "10Y-2Y", "BAMLH0A0HYM2": "HY OAS", "BAMLC0A0CM": "IG OAS",
    "DTWEXBGS": "Broad USD", "DCOILWTICO": "WTI", "NFCI": "Chicago FCI (weekly)",
    "WALCL": "Fed balance sheet (weekly)", "WTREGEN": "TGA (weekly)", "RRPONTSYD": "ON RRP",
}
FRED_VINTAGE = {
    "CPIAUCSL": "CPI", "CPILFESL": "Core CPI", "PCEPI": "PCE", "PCEPILFE": "Core PCE",
    "UNRATE": "Unemployment", "PAYEMS": "Nonfarm Payroll", "GDPC1": "Real GDP",
    "INDPRO": "Industrial Production", "M2SL": "M2", "ICSA": "Initial Claims",
    "RSAFS": "Retail Sales", "DGORDER": "Durable Goods Orders",
}

# CFTC TFF(Futures Only) 계약 코드
CFTC = {
    "13874A": "S&P500 E-mini", "209742": "Nasdaq100 E-mini", "239742": "Russell2000 E-mini",
    "043602": "UST 10Y", "042601": "UST 2Y", "098662": "US Dollar Index",
    "13874I": "S&P Tech E-mini", "13874C": "S&P Fin E-mini", "138749": "S&P Energy E-mini",
    "13874E": "S&P Health E-mini", "13874F": "S&P Ind E-mini", "13874J": "S&P Util E-mini",
    "138747": "S&P ConsDisc E-mini", "138748": "S&P ConsStaples E-mini", "13874H": "S&P Materials E-mini",
}

# 한국 업종 대표 종목(시총 상위·업종 대표성). KIS 종목별 투자자 수급을 업종별로 합산해 '한국 업종 자금흐름'을 만든다.
# 업종 ETF(위 YAHOO)와 짝을 이룬다. 구성은 고정(생존편향: 현재 대형주만 포함 — 문서에 명시).
KR_SECTOR_STOCKS = {
    "Semis": {"005930": "삼성전자", "000660": "SK하이닉스", "042700": "한미반도체"},
    "Auto": {"005380": "현대차", "000270": "기아", "012330": "현대모비스"},
    "Banks": {"105560": "KB금융", "055550": "신한지주", "086790": "하나금융지주", "316140": "우리금융지주"},
    "Healthcare": {"207940": "삼성바이오로직스", "068270": "셀트리온", "196170": "알테오젠"},
    "Defense": {"012450": "한화에어로스페이스", "079550": "LIG넥스원", "047810": "한국항공우주", "064350": "현대로템"},
    "Shipbuilding": {"329180": "HD현대중공업", "042660": "한화오션", "010140": "삼성중공업", "009540": "HD한국조선해양"},
    "EnergyChem": {"051910": "LG화학", "011170": "롯데케미칼", "009830": "한화솔루션"},
    "Battery": {"373220": "LG에너지솔루션", "006400": "삼성SDI", "003670": "포스코퓨처엠", "247540": "에코프로비엠"},
}
