# DESIGN v0 — 합의 전 초안 (코드 없음)

작성일 2026-10-13. 상태: **사용자 합의 대기**. 아래 "검증됨"은 이 날 실제로 호출해 본 결과, "미검증"은 지식·검색 기반이라 구현 전 재확인 필요.

## 1. 데이터 확보 가능성 (가장 먼저 요청된 조사)

| 데이터 | 판정 | 근거 / 대안 |
|---|---|---|
| 미국 금리·실질금리·BEI·스프레드·FCI·VIX·유동성(WALCL/RRP/TGA)·M2·CPI/PCE/고용·GDP·WTI·달러지수 | **가능 (FRED)** | `DGS10` 응답 확인(1962~2026-10-08). 단 csv 엔드포인트가 간헐적으로 타임아웃 → **공식 API + 무료 키 필요**, 재시도·캐시 필수 |
| 발표시점(vintage) 데이터 | **일부 가능 (ALFRED)** | 이 날 호출은 타임아웃으로 미검증. API 키로 `realtime_start/end` 사용. 수치형 경제지표만, 시장가격은 해당 없음 |
| ISM Manufacturing/Services, PMI | **무료 장기 이력 불가 (미검증, 알려진 제약)** | FRED에서 ISM 계열은 삭제됨. V1은 대체: 지역 연준 지수(Empire/Philly), 산업생산, 신규실업수당, 신규주문 등으로 Growth 축 구성 |
| 시장가격(섹터 ETF, SPY, SOXX, DXY 대용 등) | **가능 (Yahoo chart)** | `XLK` max 이력 응답 확인. 비공식 API라 깨질 수 있음 → 백업 소스 필요. 조정종가 사용 |
| XLC(2018~), XLRE(2015~) | **이력 짧음** | 2005~ 백테스트에선 구성종목 지수·대체(IYZ, IYR 등)로 이어붙이거나 해당 구간 결측 처리 |
| **ETF Fund Flow (장기)** | **무료로는 불가** | 일별 설정/환매 이력은 유료(QUODD, Xignite, AWS Marketplace 2017~ 등). 대안: 발행사 일별 shares outstanding × NAV 변화로 추정 — 발행사 사이트별 스크래핑, 이력 대부분 수년 이내. **V1 Capital Flow에서 ETF Flow 제외 권고** |
| CFTC 포지셔닝 | **가능, 2006-06~** | Socrata API(TFF) 응답 확인, 첫 레코드 2006-06-13. 주간(화요일 기준, 금요일 발표) → 발표일 기준 시차 반영 필수 |
| 옵션 포지셔닝 (Put/Call, GEX) | **무료 장기 불가** | CBOE 일부 Put/Call만 가능(미검증). V1 제외 |
| 기관 보유 변화 (13F) | **가능하나 분기·45일 지연** | SEC EDGAR 무료. 섹터 회전의 조기 탐지엔 부적합 → V1 제외 |
| Market breadth | **자체 계산** | 구성종목 가격 필요 → 생존편향 문제. V1은 섹터 ETF 간 breadth(상승 섹터 비율)만 |
| MOVE | 미검증 | FRED 없음. Yahoo `^MOVE` 시도 필요 |
| **한국 외국인/기관/개인 수급 (장기)** | **막힘 — 핵심 리스크** | 이 날 확인: KRX 데이터포털 직접 호출 **403**, 네이버 `investorDealTrendDay` **410(폐지)**. 잔여 경로: ① KRX 로그인 세션 기반(pykrx 계열, 계정 필요·약관 제약) ② KRX 정보데이터시스템 화면 CSV 수동 다운로드(장기 일괄 가능, 재배포 금지) ③ KRX Open API 키(제공 범위 확인 필요) ④ 증권사 API(한투 등, 계좌 필요, 이력 제한). **어느 경로로 2005~ 일별 투자자별 순매수를 받을 수 있는지 사용자 선택 필요** |
| 외국인 KOSPI200 선물 수급 | **위와 동일 경로, 이력 더 불확실** | V1 제외 후보 |
| KOSPI/KOSDAQ/KOSPI200/USDKRW | **가능** | Yahoo(`^KS11`, `^KQ11`, `KRW=X`). KOSPI200은 Yahoo 불안정 시 KRX 경로 |
| 한국 업종 지수 | **KRX 경로 의존** | 대체로 업종 대표 ETF(KODEX/TIGER, 2002~ 상장 시점별)로 가격만 추적 |
| 한국은행 ECOS | 가능(무료 키) | 금리·환율·통화량 보조 |

**결론:** 가격·매크로·유동성·CFTC는 2005/2006~ 확보 가능. **ETF 자금흐름과 기관 포지셔닝은 무료로 장기 확보 불가**, **한국 수급은 접근 경로 결정 전까지 가정 금지.** 따라서 V1의 "Capital Flow Score"는 Relative Strength + Volume + Breadth(+CFTC)로 정의하고, ETF Flow·Options·13F는 데이터 확보 후 추가하는 V2 항목으로 돌린다.

## 2. 아키텍처 (제안)

```
수집(src/data) → Parquet(data/parquet, 원천별 as-of 컬럼 포함)
  → Feature(src/features/*) → Regime 분류 → Backtest(backtest/*)
  → Score(가중치는 백테스트 결과 파일에서 로드) → Dashboard
```
- 저장: Parquet + DuckDB(쿼리). 홈서버 일 1회 크론 수집, 결과는 `site/data/*.json`.
- 대시보드: 요구 문서는 Streamlit 우선. 기존 관례(Caddy 컨테이너 + 전용 포트 **8087**)와 fermi 방식(Streamlit 8502 + Caddy)을 따른다. 정적 JSON+HTML(tenbagger 방식)이 더 가벼우나 인터랙티브 analog 검색엔 Streamlit이 유리.
- 모든 feature는 `available_at`(실제 정보 이용 가능 시각) 컬럼을 가진다. 백테스트는 `available_at <= t`인 값만 사용.

## 3. Feature (macro 예시, 모든 시리즈 공통)
level, 1/5/20/60D 변화, 기울기(회귀), 롤링 percentile·z-score(**확장윈도우, 미래 표본 미포함**). 월간 지표는 발표일에 step-hold. Economic Surprise는 컨센서스 무료 이력이 없어 **V1 제외**.

## 4. Regime
Growth/Inflation 가속·감속 → 4분면, 통화정책(FFR 6개월 방향), 유동성(순유동성 = WALCL−TGA−RRP 변화), 달러(DXY 대용 광의 달러지수 z). 임계값은 고정 규칙(예: 부호·terciles를 확장윈도우로)으로 정의, 튜닝하지 않는다.

## 5. Backtest 방법론
- 조건 충족일 추출 → 5/20/60/120 거래일 forward return(전방 겹침 보정).
- 출력: mean, median, win rate, std, MDD, N, 신뢰구간. **겹치는 관측치 때문에 유효 표본이 N보다 작다 → block bootstrap 또는 Newey-West로 CI 계산, 독립 에피소드 수를 같이 표기.**
- Train 2005–2018 / Val 2019–2022 / Test 2023~. 규칙·임계값은 Train에서 고정, Test는 한 번만 본다.
- Lead/Lag: lag 0/1/3/5/10/20D 교차상관 + 구간별 안정성(부호 일관성). "인과"가 아닌 "선행 통계적 연관"으로만 표기.
- Analog: 현재 상태 벡터와 과거 날짜의 거리(마할라노비스), 상위 k개와 최소 거리 임계 이내 개수 + 거리 표시.

## 6. 편향·오류 위험
look-ahead(발표일·개정), 데이터 개정, ETF 생존편향(상장 폐지·신규 상장), 섹터 ETF 시작일 차이, 겹치는 수익률의 과대 유의성, 다중 비교(조건 조합을 많이 시험할수록 우연한 성과), 레짐 정의 자체의 사후 튜닝, Yahoo 비공식 API 변경.

## 7. V1 제외 권고
ETF Fund Flow, 옵션 포지셔닝, 13F, Economic Surprise, 외국인 선물 수급(경로 미정), 머신러닝 전체, 산업 하위 테마의 breadth.

## 8. 사용자 결정이 필요한 것
1. 한국 수급 경로: KRX 계정 로그인 / 화면 CSV 수동 다운로드 / Open API 키 / 증권사 API 중 무엇.
2. FRED API 키 발급(무료) 후 `.env` 제공 여부.
3. ISM 대체 방침(지역 연준 지수 등 대체 OK?).
4. 대시보드 스택: Streamlit(제안) vs 정적 JSON.
5. GitHub 레포: 이 PC엔 `gh`가 없음 → 직접 만들어 주시거나 토큰 방식 결정. 제안 이름 `macro-flow-dashboard`(private).
