#!/bin/sh
# 수동 배포 — 이 폴더의 현재 파일 그대로 빌드·재기동한다. GitHub는 이력 보관용(매일 자동 커밋·push) — 이 스크립트는 git을 건드리지 않는다.
#   사용: cd ~/macro-flow-dashboard && ./deploy.sh
set -e
cd "$(dirname "$0")"
mkdir -p logs data
exec 9>logs/.lock
flock -w 900 9 || { echo "일일 갱신이 15분 넘게 실행 중 - 배포 중단"; exit 1; }
echo "$(date '+%F %T') 배포 시작"
docker compose up -d --build
# Caddyfile·compose 는 단일 파일 바인드 마운트라 편집기가 파일(inode)을 바꾸면 컨테이너가 옛 내용을 본다 → 바뀐 경우에만 caddy 재생성
SUM=$(cat Caddyfile docker-compose.yml | md5sum | cut -d' ' -f1)
if [ "$SUM" != "$(cat logs/.deploy_sum 2>/dev/null)" ]; then
    docker compose up -d --force-recreate caddy
    echo "$SUM" > logs/.deploy_sum
fi
sleep 8
docker compose ps --format '{{.Name}}  {{.Status}}'
echo "$(date '+%F %T') 배포 완료"
