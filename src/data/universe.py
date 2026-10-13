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
    "URA": "Uranium/Nuclear", "COPX": "Copper Miners", "XME": "Metals&Mining", "SLX": "Steel",
    # 매크로 가격
    "^VIX": "VIX", "^MOVE": "MOVE", "DX-Y.NYB": "DXY", "CL=F": "WTI", "GC=F": "Gold",
    "^TNX": "US10Y(yahoo)", "HYG": "HY bond ETF", "LQD": "IG bond ETF", "TLT": "20Y+ Treasury",
    # 한국
    "^KS11": "KOSPI", "^KQ11": "KOSDAQ", "^KS200": "KOSPI200", "KRW=X": "USDKRW",
    "005930.KS": "Samsung", "000660.KS": "SK hynix",
}

# FRED 시리즈. 월/분기 지표는 vintage(발표시점)로 받는다.
FRED_DAILY = {
    "DFF": "Fed Funds", "DGS2": "UST 2Y", "DGS10": "UST 10Y", "DFII10": "10Y Real Yield",
    "T10YIE": "10Y Breakeven", "T10Y2Y": "10Y-2Y", "BAMLH0A0HYG2": "HY OAS", "BAMLC0A0CM": "IG OAS",
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
