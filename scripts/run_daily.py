"""매일 자동 갱신 오케스트레이터 — 사람·AI 개입 없이 돌아가도록 설계. 컨테이너 안에서 실행한다.

  python scripts/run_daily.py --mode us   # 수집(미국·거시·공시) 후 전체 파이프라인
  python scripts/run_daily.py --mode kr   # 수집(한국 수급) 후 전체 파이프라인

원칙
  * 단계마다 시간 제한·재시도(수집)·실패 기록. 치명적 단계가 실패하면 뒤 단계를 건너뛰고(어긋난 입력 방지) 이전 결과를 그대로 둔다.
  * 결과는 data/status.json(+status_history.jsonl)에 남기고 대시보드가 '자동 갱신 상태'로 보여준다.
  * 실패·데이터 정체 시 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 .env에 있으면 알림을 보낸다(없으면 로그만).
  * 전향 검증 장부(복구 불가)는 매 실행 뒤 날짜별로 백업한다."""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MACRO_DATA_DIR", ROOT / "data"))
PY = sys.executable
COLLECT = {"us": ["yahoo", "cftc", "fred", "ssga", "sec", "cloud", "twse", "debt"], "kr": ["kis", "kis_stocks"]}
# (이름, 명령, 치명 여부, 제한시간 초)
PIPELINE = [("build_features", ["scripts/build_features.py"], True, 900), ("run_backtests", ["scripts/run_backtests.py"], False, 900),
            ("run_ai", ["scripts/run_ai.py"], False, 600), ("run_scoring", ["scripts/run_scoring.py"], True, 900),
            ("build_views", ["scripts/build_views.py"], True, 300)]


def now():
    return dt.datetime.now(dt.timezone.utc)


def run(cmd, timeout):
    t = time.time()
    try:
        p = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return p.returncode == 0, time.time() - t, (p.stdout + p.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return False, time.time() - t, f"시간 초과({timeout}s)"


def notify(text):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return False
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": text[:900]}, timeout=15)
        return True
    except Exception:
        return False


def collect_health(sources):
    """coverage.json에서 방금 수집한 소스의 오류 현황. (치명 오류 소스, 시리즈 오류 비율이 큰 소스)"""
    try:
        cov = json.loads((DATA / "coverage.json").read_text(encoding="utf-8"))
    except Exception:
        return ["coverage.json 읽기 실패"], []
    hard, soft = [], []
    for s in sources:
        v = cov.get(s)
        if v is None:
            hard.append(f"{s}: 결과 없음")
            continue
        series = v.get("series", {})
        n_err = sum("error" in x for x in series.values())
        if v.get("error") or (series and n_err == len(series)):
            hard.append(f"{s}: {v.get('error') or '전 시리즈 오류'}")
        elif series and n_err / len(series) > 0.3:
            soft.append(f"{s}: 시리즈 {n_err}/{len(series)} 오류")
    return hard, soft


def freshness():
    try:
        m = json.loads((DATA / "views" / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}, ["views/meta.json 없음"]
    today = now().date()
    problems = []
    age = lambda s: (today - dt.date.fromisoformat(s[:10])).days  # noqa: E731
    if age(m["us_close"]) > 4:
        problems.append(f"미국 마감 데이터가 {age(m['us_close'])}일 전({m['us_close']})에서 멈춤")
    if age(m["kis_last"]) > 5:
        problems.append(f"한국 수급이 {age(m['kis_last'])}일 전({m['kis_last']})에서 멈춤")
    return m, problems


def backup_ledger():
    src = DATA / "ledger" / "ledger.jsonl"
    if not src.exists():
        return
    d = DATA / "backups"
    d.mkdir(exist_ok=True)
    shutil.copy2(src, d / f"ledger_{now():%Y%m%d}.jsonl")
    for old in sorted(d.glob("ledger_*.jsonl"))[:-30]:
        old.unlink()


def trim_log():
    p = ROOT / "logs" / "daily.log"
    if p.exists() and p.stat().st_size > 2_000_000:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-4000:]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["us", "kr"], required=True)
    args = ap.parse_args()
    started = now()
    steps, warnings, ok = [], [], True
    print(f"[{started:%F %T}Z] daily update start mode={args.mode}", flush=True)

    # 1) 수집: 실패 시 최대 2회 재시도(네트워크·일시 오류)
    srcs = COLLECT[args.mode]
    for attempt in range(3):
        good, sec, tail = run(["scripts/collect.py", *srcs], 1500)
        hard, soft = collect_health(srcs)
        if good and not hard:
            break
        print(f"  collect 시도 {attempt + 1} 실패: {hard or tail[-200:]}", flush=True)
        if attempt < 2:
            time.sleep(90)
    steps.append({"name": f"collect({args.mode})", "ok": bool(good and not hard), "seconds": round(sec), "tail": "" if good and not hard else (str(hard) + tail)[-300:]})
    warnings += soft + ([f"수집 실패(이전 데이터로 계속): {hard}"] if hard else [])

    # 2) 분석 파이프라인: 치명 단계 실패 시 중단
    halted = False
    for name, cmd, critical, tmo in PIPELINE:
        if halted:
            steps.append({"name": name, "ok": False, "seconds": 0, "tail": "이전 치명 단계 실패로 건너뜀"})
            continue
        good, sec, tail = run(cmd, tmo)
        steps.append({"name": name, "ok": good, "seconds": round(sec), "tail": "" if good else tail})
        print(f"  {name}: {'ok' if good else 'FAIL'} {sec:.0f}s", flush=True)
        if not good:
            (warnings if not critical else warnings).append(f"{name} 실패")
            if critical:
                halted, ok = True, False

    backup_ledger()
    meta, problems = freshness()
    warnings += problems
    finished = now()
    status = {"mode": args.mode, "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
              "duration_s": round((finished - started).total_seconds()), "ok": ok and not problems, "warnings": warnings, "steps": steps}
    (DATA / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    with (DATA / "status_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: status[k] for k in ("mode", "started", "duration_s", "ok", "warnings")}, ensure_ascii=False) + "\n")
    hist = (DATA / "status_history.jsonl").read_text(encoding="utf-8").splitlines()
    if len(hist) > 400:
        (DATA / "status_history.jsonl").write_text("\n".join(hist[-300:]) + "\n", encoding="utf-8")
    if not status["ok"]:
        sent = notify("대시보드 자동 갱신 문제 (%s): %s" % (args.mode, "; ".join(warnings)[:700]))
        print("  알림 전송:", sent, flush=True)
    trim_log()
    print(f"[{finished:%F %T}Z] done ok={status['ok']} warnings={warnings}", flush=True)
    sys.exit(0 if status["ok"] else 1)


if __name__ == "__main__":
    main()
