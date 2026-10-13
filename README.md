# Macro & Capital Flow Dashboard

외부 시장환경(Macro → Liquidity → Wall Street Positioning → Sector Rotation → Korea Transmission)을
백테스트로 검증하고, 현재 시장을 과거 유사 국면과 비교하는 프로젝트. 펀더멘털은 의도적으로 제외.

- 상태: Phase 1 (데이터 파이프라인). Yahoo·CFTC·KIS 수집 가동, FRED는 API 키 대기. 설계: `docs/DESIGN.md`
- 홈서버 포트: 8087 (LAN, Caddy basic_auth)
- 원칙: 설명 가능성, look-ahead bias 금지, 데이터 확보 불가한 feature는 가정하지 않는다.
