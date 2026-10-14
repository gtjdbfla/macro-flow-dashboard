#!/bin/sh
# 서버에서 크론으로 돈다 (5,35분). 원격(GitHub) 변경을 받아 필요한 만큼만 반영한다.
#   5,35 * * * * /home/yulimseo/macro-flow-dashboard/deploy.sh >> /home/yulimseo/macro-flow-dashboard/logs/deploy.log 2>&1
#
# 앱은 살아있는 파이썬 프로세스라 .py가 바뀌면 재빌드·재기동이 필요하다(git pull만으로는 반영 안 됨).

set -e
cd "$(dirname "$0")"
mkdir -p logs data

# 일일 파이프라인(scripts/run_daily.py)이 같은 잠금을 잡고 있으면 재빌드로 실행 중인 컨테이너를 끊지 않도록 이번 회차는 건너뛴다(30분 뒤 재시도).
exec 9>logs/.lock
if ! flock -n 9; then
    echo "$(date '+%F %T') 일일 갱신 실행 중 - 배포 건너뜀"
    exit 0
fi

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
# Caddyfile은 단일 파일 바인드 마운트라 git reset이 파일(inode)을 바꾸면 컨테이너가 옛 내용을 계속 본다
# (2026-10-13 basic_auth 제거가 반영 안 된 원인). reload로는 부족하고 컨테이너를 다시 만들어야 한다.
case "$CHANGED" in
    *Caddyfile*|all) docker compose up -d --force-recreate caddy ;;
esac
echo "$(date '+%F %T') 배포 완료"
