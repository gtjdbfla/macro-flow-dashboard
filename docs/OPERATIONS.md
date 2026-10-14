# 운영 (사람·AI 개입 없이 매일 자동 갱신)

## 구조
- **서버 크론 3줄** (`scripts/install_cron.sh`로 설치, 멱등): 코드 배포(5·35분마다 `deploy.sh`), 한국 수급 갱신(월~금 07:40 UTC), 미국·거시·공시 갱신(월~금 22:30 UTC).
- 갱신 = `docker compose exec`로 컨테이너 안에서 **한 번** `scripts/run_daily.py --mode kr|us` 실행. 안에서 순서대로:
  수집(재시도 2회) → build_features → run_backtests → run_ai → run_scoring → build_views → 장부 백업 → 신선도 점검 → 상태 기록.
- 치명 단계(build_features·run_scoring·build_views)가 실패하면 뒤 단계를 건너뛰고 **이전 결과를 그대로 유지**(어긋난 입력으로 화면이 깨지지 않게). 수집이 실패해도 이전 데이터로 계속하고 경고로 남긴다.
- `flock`(logs/.lock)이 `deploy.sh`와 갱신을 서로 막는다 — 갱신 중에는 재빌드가 끼어들지 않고, 배포 중에는 갱신이 건너뛰어진다(다음 회차에 실행).
- 결과: `data/status.json`(마지막 실행), `data/status_history.jsonl`(최근 300회), 대시보드 '근거·방법 > 자동 갱신 상태', 문제가 있으면 홈 상단 경고. 로그 `logs/daily.log`.
- 전향 검증 장부 `data/ledger/ledger.jsonl`은 복구가 불가능해 매 실행 뒤 `data/backups/ledger_YYYYMMDD.jsonl`로 백업(30일).

## 확인·수동 실행
```
ssh homeserver 'tail -n 30 ~/macro-flow-dashboard/logs/daily.log'
ssh homeserver 'cat ~/macro-flow-dashboard/data/status.json | head -40'
# 수동 실행(잠금 포함)
ssh homeserver 'cd ~/macro-flow-dashboard && flock -n logs/.lock docker compose exec -T macro-flow-dashboard python scripts/run_daily.py --mode us'
```

## 알림(선택)
`~/macro-flow-dashboard/.env`에 `TELEGRAM_BOT_TOKEN=...`, `TELEGRAM_CHAT_ID=...`를 넣고 `docker compose up -d`로 반영하면, 갱신 실패·데이터 정체(미국 4일, 한국 5일 이상 정지) 시 텔레그램으로 알린다. 없으면 로그·대시보드 경고만.

## 알려진 함정
- 크론 줄에 `docker compose exec A && python B`처럼 이어 쓰면 B는 호스트에서 실행돼 깨진다(이 프로젝트에서 실제로 겪음). 반드시 `run_daily.py` 한 번으로.
- 서버에서 직접 `git reset`을 하면 `deploy.sh`가 변경을 못 알아채 재빌드하지 않는다 — 배포는 push 후 `deploy.sh`에 맡기거나 `docker compose up -d --build`.
- KIS 종목별 수급은 한국시간 15:40 이전에는 당일 조회가 거부된다(코드가 전일로 대체). KIS 앱키는 발급 1년 후 만료(2027-09경) — 만료 30일 전 포털에서 갱신하고 서버 `.env`를 교체.
- 대만 월매출은 최신월 스냅샷만 제공돼 수집이 멈추면 그 달 이력이 비게 된다.
