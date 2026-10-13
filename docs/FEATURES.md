# Feature 정의 (Phase 2)

빌드: `python scripts/build_features.py` → `data/parquet/features_{macro,liquidity,risk,sector,flow,korea,state}.parquet`,
컬럼별 실제 시작일은 `data/features_meta.json`. 인덱스 = SPY 거래일(2005-01-03~), 행 t = "미국 마감(t) 시점에 알 수 있던 정보".

## 규칙
- **발표시점**: 월/분기 지표는 ALFRED vintage의 *최초 공표값*을 `available_at`에 배치. 최초 공표가 date+3×기본지연보다 늦으면(DB 적재 이전 과거분) date+기본지연으로 근사(`approx`). 일/주 FRED는 지연일 고정(H.15·H.4.1 +1일, NFCI +5일), CFTC +3일.
- **percentile/z**: expanding(과거만). 일별 최소 252, 월별 60, 주간 CFTC 104.
- **검증**: `pytest tests` — T까지 정보만으로 만든 feature = 전체 feature의 T 이전 행. 2012/2020/2024 세 시점 통과. 일부러 심은 누수(shift(-5))가 검출됨을 확인.
- 가격 *수준*은 쓰지 않는다(조정종가의 소급 조정). 수익률·비율 변화만.

## 레이어
| 레이어 | 내용 |
|---|---|
| macro | 금리·실질금리·BEI·곡선·WTI·DXY 일별(수준/1·5·20·60D 변화/기울기/z/rank), CPI·PCE·고용·산업생산·소매·M2·내구재(yoy, 3M연율, yoy 3M변화), 실업률, GDP, 청구건수 |
| liquidity | 순유동성(Fed 자산 − TGA − 역RP), 연준자산, TGA, NFCI, 광의달러 |
| risk | VIX, MOVE, HYG/LQD, HY·IG OAS(FRED 2023-10~만), SPY 낙폭·실현변동성 |
| sector | 11섹터+17산업 ETF: 수익률·SPY 대비 RS(1~120D)·12-1 모멘텀·RS 가속·거래량 z, 섹터 breadth |
| flow | CFTC TFF 투기(levmoney)/자산운용/딜러 순포지션(OI 대비), 4주 변화, rank. 섹터 E-mini는 2017~ |
| korea | KOSPI/KOSDAQ/KODEX200 수익률, USDKRW, 업종 ETF RS, 투자자별(외국인·기관·개인) 순매수 누적·z |
| state | Growth/Inflation(4분면), Policy, Liquidity, Dollar — 규칙 고정 |

## 한계 (알려진 것)
- 한국 t일 장은 미국 t일 개장 전에 끝난다 → 행 t에 한국 t일이 이미 들어 있다. US→KR 전달은 Phase 3에서 한국을 shift(-1)로 정렬해 본다.
- 업종 ETF 시작일: 방산 2023-07, 조선 2023-10, 2차전지 2018-09 (역사 백테스트는 해당 구간만).
- Economic Surprise, ETF Flow, 옵션, 13F 없음(데이터 없음).
- Regime 라벨은 단순 부호 규칙이라 경계에서 자주 뒤집힌다(히스테리시스 없음). 백테스트에서 안정성을 보고 판단.
