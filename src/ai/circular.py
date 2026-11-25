"""순환 구조(공급자·클라우드가 AI 기업에 돈을 대고 그 돈이 다시 돌아오는 연결고리) — 회사가 공시한 값만 모은다. 금액 단위 억 달러.
판정 없이 표시만 한다(docs/AI_PROFITABILITY.md 2026-11-25). 분기마다 새 10-Q가 나오면 손으로 갱신한다(일일 루틴 점검 대상)."""
import pandas as pd

MSFT_10Q = "https://www.sec.gov/Archives/edgar/data/789019/000119312526436133/msft-20260930.htm"
AMZN_10Q = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000040/amzn-20260930.htm"
GOOG_10Q = "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000084/goog-20260930.htm"
ORCL_10Q = "https://www.sec.gov/Archives/edgar/data/1341439/000119312526389274/orcl-20260831.htm"
ORCL_NEWS = "https://siliconangle.com/2026/06/10/oracles-ai-investments-spook-investors-despite-another-impressive-earnings-revenue-beat/"

# (회사, 기준, 항목, 값 설명, 출처). 값은 원문 숫자를 옮긴 것이다.
FACTS = (
    ("Microsoft ↔ OpenAI", "2026-09-30", "지분", "지분법 투자, 완전 전환 기준 약 24%", MSFT_10Q),
    ("Microsoft ↔ OpenAI", "2026-09-30", "투자 약정", "약정 130억 중 120억 집행", MSFT_10Q),
    ("Microsoft ↔ OpenAI", "2026-07~09", "OpenAI에서 받은 매출", "100억(수익 배분 포함). 같은 분기 Microsoft 매출 922.8억의 약 11%", MSFT_10Q),
    ("Microsoft ↔ OpenAI", "2026-09-30", "OpenAI 미수금", "72억(2026-06-30 60억)", MSFT_10Q),
    ("Microsoft ↔ OpenAI", "2026-07~09", "지분법 손익", "순이익 +2.07억(전년 같은 분기 −31억)", MSFT_10Q),
    ("Amazon → Anthropic", "2023~2025", "전환사채 투자", "80억 달러(2023년 3분기~2025년 4분기)", AMZN_10Q),
    ("Amazon → Anthropic", "2026-04~06", "우선주 투자", "Series G 50억 + Series H 50억", AMZN_10Q),
    ("Amazon → Anthropic", "2026-09-30", "자금 지원 한도", "최대 200억 중 150억 남음. 기업공개 후 30개월까지. AWS가 컴퓨팅 용량 납품 이정표를 달성할 때마다 Anthropic이 쓸 수 있게 열림", AMZN_10Q),
    ("Amazon → Anthropic", "2026-09-30", "장부가(공정가치)", "전환사채 1,462억(2025-12 458억), 우선주 925억(148억)", AMZN_10Q),
    ("Amazon → Anthropic", "2026-01~09", "평가이익(영업외수익)", "우선주 628억 상향 조정. 같은 기간 세전이익 1,475.6억의 약 43%, 순이익 1,135.9억(전년 564.8억)", AMZN_10Q),
    ("Alphabet", "2026-01~09", "비시장성 지분 평가이익", "순 1,163억, 지분증권 전체 손익 1,272.8억(순이익 2,012.4억). 10-Q는 Anthropic을 이름으로 밝히지 않음", GOOG_10Q),
    ("Alphabet", "2026-07~09", "지분증권 손익", "−86.6억(비시장성 +30.3억, 시장성 −124.4억)", GOOG_10Q),
    ("Alphabet", "2026-09-30", "데이터센터 보증", "제3자 데이터센터 임차료·전력비 지급 보증과 조건부 추가 출자 약정을 파생상품(3단계)으로 기록", GOOG_10Q),
    ("Oracle", "2026-08-31", "남은 이행 의무(RPO)", "6,640억(1년 전 4,550억). 10-Q는 고객 이름을 밝히지 않음", ORCL_10Q),
    ("Oracle", "2026-05-31", "OpenAI 비중(보도)", "BofA 분석가: RPO의 50% 넘게 OpenAI Group(2차 출처, 공시 아님·미확인)", ORCL_NEWS),
)


def facts() -> pd.DataFrame:
    return pd.DataFrame(FACTS, columns=["고리", "기준", "항목", "내용", "출처"])


def amazon_markup_share(markup=628.0, pretax=1475.6) -> float:
    """Amazon 2026-01~09 Anthropic 평가이익이 세전이익에서 차지하는 비중."""
    return markup / pretax
