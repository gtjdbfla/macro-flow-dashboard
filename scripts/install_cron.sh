#!/bin/sh
# 서버에서 한 번 실행: 이 프로젝트의 크론을 멱등하게 설치한다(기존 macro-flow 줄은 교체, 다른 프로젝트 줄은 건드리지 않음).
#   sh ~/macro-flow-dashboard/scripts/install_cron.sh
#
# 일정(UTC)
#   07:40 월~금   한국 장 마감 후(16:40 KST) 한국 수급 갱신 (kr)
#   22:30 월~금   미국 마감 후 가격·거시·ETF·공시·채권 갱신 (us), 대만 월매출은 매월 10~12일만
#   일 09:00      백테스트·검증 통계까지 전체 재계산 (weekly)
#   3시간마다     1·4·7·10월 15~31일: SEC·클라우드·채권 새 공시만 확인 (filings) — Big4 실적 발표일 반영 지연 최소화
# 크론 필드 주의: 일(day-of-month)과 요일을 둘 다 지정하면 OR로 해석되므로 filings 줄은 요일을 *로 둔다(주말에도 가볍게 돈다).
# 실행은 컨테이너 안에서 한 번의 docker exec(단계 체인을 크론 줄에 풀어 쓰면 첫 명령만 컨테이너에서 돌고 나머지는 호스트에서 돌아 깨진다).
# flock: deploy.sh도 같은 잠금 -> 갱신 중엔 재빌드가 끼어들지 않는다.
set -e
D=/home/yulimseo/macro-flow-dashboard
mkdir -p "$D/logs"
TMP=$(mktemp)
crontab -l 2>/dev/null | grep -v "macro-flow-dashboard" > "$TMP" || true
RUN="cd $D && /usr/bin/flock -n $D/logs/.lock /usr/bin/docker compose exec -T macro-flow-dashboard python scripts/run_daily.py"
cat >> "$TMP" <<EOF
5,35 * * * * $D/deploy.sh >> $D/logs/deploy.log 2>&1
40 7 * * 1-5 $RUN --mode kr >> logs/daily.log 2>&1
30 22 * * 1-5 $RUN --mode us >> logs/daily.log 2>&1
0 9 * * 0 $RUN --mode weekly >> logs/daily.log 2>&1
15 */3 15-31 1,4,7,10 * $RUN --mode filings >> logs/daily.log 2>&1
EOF
crontab "$TMP"
rm -f "$TMP"
crontab -l | grep macro-flow-dashboard
