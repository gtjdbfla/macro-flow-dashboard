#!/bin/sh
# 서버에서 한 번 실행: 이 프로젝트의 크론을 멱등하게 설치한다(기존 macro-flow 줄은 교체, 다른 프로젝트 줄은 건드리지 않음).
#   sh ~/macro-flow-dashboard/scripts/install_cron.sh
#
# 일정(UTC):  07:40 월~금 = 한국 장 마감 후(16:40 KST) 수급 수집  /  22:30 월~금 = 미국 마감 후 수집(한국시간 화~토 07:30)
# 실행은 컨테이너 안에서 한 번의 docker exec로 — 단계 체인을 크론 줄에 풀어 쓰면 첫 명령만 컨테이너에서 돌고 나머지는 호스트에서 돌아 깨진다.
# flock: 같은 잠금 파일을 deploy.sh도 쓴다 -> 파이프라인이 도는 동안 재빌드가 끼어들지 않는다.
set -e
D=/home/yulimseo/macro-flow-dashboard
mkdir -p "$D/logs"
TMP=$(mktemp)
crontab -l 2>/dev/null | grep -v "macro-flow-dashboard" > "$TMP" || true
cat >> "$TMP" <<EOF
5,35 * * * * $D/deploy.sh >> $D/logs/deploy.log 2>&1
40 7 * * 1-5 cd $D && /usr/bin/flock -n $D/logs/.lock /usr/bin/docker compose exec -T macro-flow-dashboard python scripts/run_daily.py --mode kr >> logs/daily.log 2>&1
30 22 * * 1-5 cd $D && /usr/bin/flock -n $D/logs/.lock /usr/bin/docker compose exec -T macro-flow-dashboard python scripts/run_daily.py --mode us >> logs/daily.log 2>&1
EOF
crontab "$TMP"
rm -f "$TMP"
crontab -l | grep macro-flow-dashboard
