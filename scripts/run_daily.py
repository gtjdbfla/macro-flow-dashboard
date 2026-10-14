"""자동 갱신 오케스트레이터 — 사람·AI 개입 없이 돌아가도록 설계. 컨테이너 안에서 실행한다.

  --mode kr        한국 장 마감 후: 한국 수급 수집 -> 분석 갱신           (월~금 07:40 UTC)
  --mode us        미국 마감 후: 가격·거시·ETF·공시·채권 수집 -> 분석 갱신   (월~금 22:30 UTC) 대만 월매출은 매월 10~12일에만
  --mode weekly    백테스트·검증 통계 포함 전체 재계산(수집 없음)           (일요일 09:00 UTC)
  --mode filings   실적 시즌 2주: SEC·클라우드·채권만 가볍게 확인, 새 공시가 있을 때만 분석 갱신 (3시간마다)

원칙
  * 백테스트는 주 1회: 하루 새 데이터가 늘어도 결과가 거의 안 변하고 실행만 길어진다(없으면 처음 한 번 계산).
  * 단계마다 시간 제한·재시도(수집)·실패 기록. 치명 단계가 실패하면 뒤 단계를 건너뛰고 이전 결과를 유지(어긋난 입력 방지).
  * 결과는 data/status.json(+status_history.jsonl)에 남고 대시보드가 보여준다. 실패·정체 시 TELEGRAM_* 설정이 있으면 알림.
  * 전향 검증 장부(복구 불가)는 매 실행 뒤 날짜별로 백업."""
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
PARQ = DATA / "parquet"
PY = sys.executable
COLLECT = {"us": ["yahoo", "cftc", "fred", "ssga", "sec", "cloud", "debt"], "kr": ["kis", "kis_stocks"],
           "filings": ["sec", "cloud", "debt"], "weekly": []}
FEATURES = ("build_features", ["scripts/build_features.py"], True, 900)
BACKTESTS = ("run_backtests", ["scripts/run_backtests.py"], False, 900)
AI = ("run_ai", ["scripts/run_ai.py"], False, 600)
SCORING = ("run_scoring", ["scripts/run_scoring.py"], True, 900)
VIEWS = ("build_views", ["scripts/build_views.py"], True, 300)


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


def twse_due(today: dt.date) -> bool:
    """대만 월매출은 매월 10~12일에 나온다. 그 기간이거나, 지난달 값을 아직 못 받았으면 수집."""
    if 10 <= today.day <= 12:
        return True
    f = PARQ / "twse_monthly.parquet"
    if not f.exists():
        return True
    import pandas as pd
    last = pd.read_parquet(f, columns=["month"]).month.max().date()
    prev_month_end = today.replace(day=1) - dt.timedelta(days=1)
    return today.day >= 13 and last < prev_month_end


def filings_snapshot():
    """새 공시가 있었는지 비교하기 위한 지문(최신 공시일·행 수)."""
    import pandas as pd
    snap = {}
    for name, col in [("sec_quarterly", "filed"), ("cloud_segments", "filed"), ("debt_issuance", "filed")]:
        f = PARQ / f"{name}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=[col])
            snap[name] = (str(d[col].max()), len(d))
    return snap


def collect_health(sources):
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
    ap.add_argument("--mode", choices=list(COLLECT), required=True)
    args = ap.parse_args()
    mode = args.mode
    started = now()
    steps, warnings, ok, halted = [], [], True, False
    print(f"[{started:%F %T}Z] update start mode={mode}", flush=True)

    # 1) 수집
    srcs = list(COLLECT[mode])
    if mode == "us" and twse_due(started.date()):
        srcs.append("twse")
    before = filings_snapshot() if mode == "filings" else None
    if srcs:
        for attempt in range(3):
            good, sec, tail = run(["scripts/collect.py", *srcs], 1500)
            hard, soft = collect_health(srcs)
            if good and not hard:
                break
            print(f"  collect 시도 {attempt + 1} 실패: {hard or tail[-200:]}", flush=True)
            if attempt < 2:
                time.sleep(90 if mode != "filings" else 30)
        steps.append({"name": f"collect({','.join(srcs)})", "ok": bool(good and not hard), "seconds": round(sec),
                      "tail": "" if good and not hard else (str(hard) + tail)[-300:]})
        warnings += soft + ([f"수집 실패(이전 데이터로 계속): {hard}"] if hard else [])

    # 2) 분석 단계 선택
    if mode == "filings":
        changed = filings_snapshot() != before
        pipeline = [AI, SCORING, VIEWS] if changed else []
        steps.append({"name": "새 공시 확인", "ok": True, "seconds": 0, "tail": "새 공시 있음 -> 분석 갱신" if changed else "새 공시 없음"})
    elif mode == "weekly":
        pipeline = [FEATURES, BACKTESTS, AI, SCORING, VIEWS]
    else:
        pipeline = [FEATURES, AI, SCORING, VIEWS]
        if not (DATA / "results" / "regime_backtest.parquet").exists():
            pipeline.insert(1, BACKTESTS)
    for name, cmd, critical, tmo in pipeline:
        if halted:
            steps.append({"name": name, "ok": False, "seconds": 0, "tail": "이전 치명 단계 실패로 건너뜀"})
            continue
        good, sec, tail = run(cmd, tmo)
        steps.append({"name": name, "ok": good, "seconds": round(sec), "tail": "" if good else tail})
        print(f"  {name}: {'ok' if good else 'FAIL'} {sec:.0f}s", flush=True)
        if not good:
            warnings.append(f"{name} 실패")
            if critical:
                halted, ok = True, False

    if mode != "filings":
        backup_ledger()
        _, problems = freshness()
        warnings += problems
    else:
        problems = []
    finished = now()
    status = {"mode": mode, "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
              "duration_s": round((finished - started).total_seconds()), "ok": ok and not problems and not (mode == "filings" and warnings and halted),
              "warnings": warnings, "steps": steps}
    (DATA / f"status_{mode}.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    if mode != "filings":  # 3시간마다 도는 가벼운 점검이 '마지막 전체 갱신' 표시를 덮지 않게 한다
        (DATA / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    skip_hist = mode == "filings" and not any(s["name"] == "새 공시 확인" and "있음" in s["tail"] for s in steps) and status["ok"]
    if not skip_hist:  # 새 공시가 없던 점검은 이력에 남기지 않는다(하루 8번 쌓이는 것 방지)
        with (DATA / "status_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: status[k] for k in ("mode", "started", "duration_s", "ok", "warnings")}, ensure_ascii=False) + "\n")
        hist = (DATA / "status_history.jsonl").read_text(encoding="utf-8").splitlines()
        if len(hist) > 400:
            (DATA / "status_history.jsonl").write_text("\n".join(hist[-300:]) + "\n", encoding="utf-8")
    if not status["ok"]:
        sent = notify("대시보드 자동 갱신 문제 (%s): %s" % (mode, "; ".join(warnings)[:700]))
        print("  알림 전송:", sent, flush=True)
    trim_log()
    print(f"[{finished:%F %T}Z] done ok={status['ok']} warnings={warnings}", flush=True)
    sys.exit(0 if status["ok"] else 1)


if __name__ == "__main__":
    main()
