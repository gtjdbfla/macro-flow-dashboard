#!/bin/sh
# 서버에서 크론으로 돈다 (5,35분). 원격(GitHub) 변경을 받아 필요한 만큼만 반영한다.
#   5,35 * * * * /home/yulimseo/macro-flow-dashboard/deploy.sh >> /home/yulimseo/macro-flow-dashboard/logs/deploy.log 2>&1
#
# 앱은 살아있는 파이썬 프로세스라 .py가 바뀌면 재빌드·재기동이 필요하다(git pull만으로는 반영 안 됨).

set -e
cd "$(dirname "$0")"
mkdir -p logs data

LOG=logs/deploy.log
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 4000 ]; then
    tail -n 2000 "$LOG" > "$LOG.keep" && cat "$LOG.keep" > "$LOG"
    rm -f "$LOG.keep"
fi

BEFORE=$(git rev-parse HEAD)
git fetch -q origin main
git reset -q --hard origin/main
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ] && docker ps --format '{{.Names}}' | grep -q '^macro-flow-dashboard$'; then
    echo "$(date '+%F %T') 변경 없음 ($AFTER)"
    exit 0
fi

CHANGED=$(git diff --name-only "$BEFORE" "$AFTER" 2>/dev/null || echo all)
echo "$(date '+%F %T') $BEFORE -> $AFTER : $CHANGED"
docker compose up -d --build
# Caddyfile은 마운트 파일 변경을 자동 감지하지 않는다.
docker exec macro-flow-caddy caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | tail -1 || true
echo "$(date '+%F %T') 배포 완료"
