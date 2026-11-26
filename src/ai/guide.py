"""투자 길잡이용 상태 스냅샷과 주간 변화. 신호·수치는 기존 계산 결과를 그대로 모으기만 하고 새 예측이나 종합 점수는 만들지 않는다.
개선 점검표(CHECKLIST)는 규칙을 docs/AI_PROFITABILITY.md(길잡이 절)에 먼저 적어 둔 예/아니오 항목이며 항목 수 합계에 예측력이 있다고 주장하지 않는다."""
import json

import pandas as pd

SNAPSHOT_MIN_DAYS = 6
LEVEL_RANK = {"긍정": 2, "강함": 2, "정상": 2, "일치": 2, "중립": 1, "보류": 0, "주의": -1, "둔화": -1, "경고": -2}

# (id, 설명, 분기 지표 이름). 최근 두 분기말의 Big4 합계를 비교한다. 전부 후행 지표라 분기 실적이 나올 때만 바뀐다.
CHECKLIST = (
    ("capex_decel", "Big4 CAPEX 증가율이 직전 분기보다 낮아졌다", "capex_yoy", "down"),
    ("capex_ocf_down", "Big4 CAPEX ÷ 영업현금흐름이 직전 분기보다 낮아졌다", "capex_ocf", "down"),
    ("fcf_up", "Big4 12개월 잉여현금흐름이 직전 분기보다 늘었다", "fcf_ttm", "up"),
    ("payback_up", "영업이익 증가액 ÷ 1년 전 CAPEX가 직전 분기보다 높아졌다", "opinc_per_capex", "up"),
)
FLAG_CHECKS = (
    ("cloud_ok", "클라우드 성장·수익성(F7)이 긍정이다", "F7", {"긍정"}),
    ("software_ok", "소프트웨어 AI 매출화(F18)가 긍정이다", "F18", {"긍정"}),
    ("nvda_cash_ok", "NVIDIA 수요의 질(F16)이 정상이다", "F16", {"정상"}),
    ("borrowers_ok", "차입 의존 기업 런웨이(F11)가 경고가 아니다", "F11", {"긍정", "정상", "중립", "주의", "보류"}),
)


def checklist(big4, flags):
    """big4: ai_history의 BIG4 행. flags: {id: level}."""
    h = big4.sort_values("period_end").dropna(subset=["capex_yoy"]).tail(2)
    items = []
    for key, text, col, direction in CHECKLIST:
        if len(h) < 2 or h[col].isna().any():
            items.append({"id": key, "text": text, "ok": None, "detail": "자료 부족"})
            continue
        prev, cur = float(h[col].iloc[0]), float(h[col].iloc[1])
        ok = cur < prev if direction == "down" else cur > prev
        items.append({"id": key, "text": text, "ok": bool(ok),
                      "detail": f"{h.period_end.iloc[0]:%Y-%m} {prev:,.3g} → {h.period_end.iloc[1]:%Y-%m} {cur:,.3g}"})
    for key, text, fid, good in FLAG_CHECKS:
        lvl = flags.get(fid)
        items.append({"id": key, "text": text, "ok": None if lvl is None else lvl in good, "detail": f"{fid} {lvl or '없음'}"})
    return items


def snapshot(date, risk, flags, big4, metrics=None, ctx=None):
    """한 날짜의 상태. risk=risk_gauge.json, flags=ai_flags.json의 flags 목록."""
    levels = {f["id"]: f["level"] for f in flags}
    macro = risk.get("macro", {})
    return {
        "date": str(date),
        "macro": {"outlook": macro.get("outlook"), "as_of": macro.get("as_of"),
                  "metrics": {k: {"state": v["state"], "value": v["value"], "change": v["change"]}
                              for k, v in macro.get("metrics", {}).items()}},
        "market": {k: risk.get(k) for k in ("as_of", "score", "level", "spx", "spx_dd", "vix", "shock", "kospi_level", "kospi_dd")},
        "flags": levels,
        "checklist": checklist(big4, levels),
        "bubble": bubble_checks(big4, levels, ctx or {}, date),
        "metrics": metrics or {},
    }


def append(path, snap):
    """같은 날짜의 스냅샷은 덮어쓴다. 장부는 날짜 오름차순 JSON Lines."""
    rows = []
    if path.exists():
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["date"] != snap["date"]] + [snap]
    rows.sort(key=lambda r: r["date"])
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return rows


def compare(rows, min_days=SNAPSHOT_MIN_DAYS):
    """가장 최근 스냅샷과 min_days 이상 전의 가장 가까운 스냅샷을 비교한다. 없으면 이전 스냅샷 중 가장 오래된 것."""
    cur = rows[-1]
    older = [r for r in rows[:-1] if (pd.Timestamp(cur["date"]) - pd.Timestamp(r["date"])).days >= min_days]
    prev = older[-1] if older else (rows[0] if len(rows) > 1 else None)
    if prev is None:
        return {"base": None, "flag_changes": [], "macro_changes": [], "checklist_changes": [], "market": {}, "bubble_changes": [], "early_changes": []}
    flag_changes = []
    for fid, lvl in cur["flags"].items():
        before = prev["flags"].get(fid)
        if before is not None and before != lvl:
            flag_changes.append({"id": fid, "from": before, "to": lvl,
                                 "direction": "개선" if LEVEL_RANK.get(lvl, 0) > LEVEL_RANK.get(before, 0) else "악화"
                                 if LEVEL_RANK.get(lvl, 0) < LEVEL_RANK.get(before, 0) else "변화"})
    macro_changes = []
    if cur["macro"]["outlook"] != prev["macro"]["outlook"]:
        macro_changes.append({"name": "종합 판정", "from": prev["macro"]["outlook"], "to": cur["macro"]["outlook"]})
    for k, v in cur["macro"]["metrics"].items():
        before = prev["macro"]["metrics"].get(k)
        if before and before["state"] != v["state"]:
            macro_changes.append({"name": k, "from": before["state"], "to": v["state"]})
    checklist_changes = []
    before_items = {i["id"]: i["ok"] for i in prev["checklist"]}
    for item in cur["checklist"]:
        if item["id"] in before_items and before_items[item["id"]] != item["ok"]:
            checklist_changes.append({"id": item["id"], "text": item["text"], "from": before_items[item["id"]], "to": item["ok"]})
    market = {}
    for k in ("score", "spx", "vix"):
        a, b = cur["market"].get(k), prev["market"].get(k)
        if a is not None and b is not None:
            market[k] = {"from": b, "to": a}
    return {"base": prev["date"], "flag_changes": flag_changes, "macro_changes": macro_changes,
            "checklist_changes": checklist_changes, "market": market,
            "bubble_changes": bubble_changes(cur.get("bubble") or {"resolution": [], "warning": []}, prev.get("bubble")),
            "early_changes": early_changes(cur.get("early"), prev.get("early"))}


def early_changes(cur, prev):
    """조기경보 신호등과 추세 상태(보유/이탈)가 바뀐 항목."""
    if not cur or not prev:
        return []
    out = []
    if cur.get("stage") and prev.get("stage") and cur["stage"] != prev["stage"]:
        out.append({"kind": "단계", "id": "시장 단계", "from": prev["stage"], "to": cur["stage"]})
    for grp, key in (("levels", "지표"), ("trend", "추세")):
        for k, v in cur.get(grp, {}).items():
            b = prev.get(grp, {}).get(k)
            if b is not None and b != v:
                out.append({"kind": key, "id": k, "from": b, "to": v})
    return out


def annotate_since(bubble, rows):
    """충족·발생(ok=True)인 점검표 항목마다 연속으로 켜진 시작일을 장부에서 찾아 since, since_kind를 붙인다.
    since_kind: 'exact'(그 전날 장부에 꺼져 있었음), 'record'(장부 시작일까지 계속 켜져 있어 그 이전부터일 수 있음), 'new'(그 전 장부에 항목이 없던 새 항목)."""
    dated = [r for r in rows if r.get("bubble")]
    for key in ("resolution", "warning"):
        for item in bubble.get(key, []):
            if item["ok"] is not True:
                continue
            since, kind = None, "record"
            for r in reversed(dated):
                prev = next((x for x in r["bubble"].get(key, []) if x["id"] == item["id"]), None)
                if prev is None:
                    kind = "new"
                    break
                if prev["ok"] is not True:
                    kind = "exact"
                    break
                since = r["date"]
            item["since"], item["since_kind"] = since, kind


def streak_start(series, test):
    """series(시간 오름차순)의 끝에서부터 test(값, 직전 값)가 연속으로 참인 구간의 시작 인덱스. 최신 값이 거짓이면 None."""
    vals = list(series.items())
    start = None
    for i in range(len(vals) - 1, 0, -1):
        cur, prev = vals[i][1], vals[i - 1][1]
        if pd.isna(cur) or pd.isna(prev) or not test(cur, prev):
            break
        start = vals[i][0]
    return start


def retro_since(bubble, ramp_all=None, support=None, released=None, adopt_up=0.03):
    """장부가 짧은 두 항목의 켜진 날을 원자료로 거슬러 계산한다. B4(AI 도입 확산, Ramp 6개월 변화 +3%p 이상): 연속 충족이 시작된 관측월.
    C1(NVIDIA 지원성 약정 ÷ 매출, 직전 분기보다 낮다): 연속 하락이 시작된 분기의 실적 발표일. 값이 없으면 건드리지 않는다."""
    items = {i["id"]: i for i in bubble.get("resolution", [])}
    b4 = items.get("B4")
    if b4 and b4["ok"] is True and ramp_all is not None and len(ramp_all.dropna()) > 6:
        s = ramp_all.dropna().sort_index()
        change = s - s.shift(6)
        flags = [bool(v >= adopt_up) for v in change.dropna()]
        idx = list(change.dropna().index)
        start = None
        for d, f in zip(reversed(idx), reversed(flags)):
            if not f:
                break
            start = d
        if start is not None:
            b4["since"], b4["since_kind"] = str(start.date()), "obs"
    c1 = items.get("C1")
    if c1 and c1["ok"] is True and support is not None and released:
        start = streak_start(support.dropna(how="all") if hasattr(support, "dropna") else support, lambda cur, prev: cur < prev)
        key = str(start.date()) if start is not None else None
        if key in released:
            c1["since"], c1["since_kind"] = released[key], "release"


# ── 거품론 점검표(docs/AI_PROFITABILITY.md, 2026-11-20 사전 등록) ─────────────────────────────────────────
COVERAGE_MAX_AGE, CDS_MAX_AGE = 200, 10
CAPEX_OCF_OK, CAPEX_OCF_DROP_PP = 0.75, 0.05          # A1(2026-11-26 개정)
RATE_20D_BP, HY_20D_BP = -25, -20                     # C5·C6: scoring/macro_outlook.py의 rates·credit 기준
W_CAPEX_OCF, W_HY_FROM_LOW_BP, W_CDS_BP = 1.0, 100, 1000   # W7·W8·W9
CAPEX_DROP = 0.10
BUBBLE_GROUPS = {"A": "현금 회수", "B": "최종 수요", "C": "금융", "D": "시장 안정(가격)"}


def _item(group, key, text, ok, detail):
    return {"group": group, "id": key, "text": text, "ok": None if ok is None else bool(ok), "detail": detail}


def _less(a, b, fmt="{:.3g}"):
    """a < b 를 (판정, 설명)으로. 값이 없으면 판정 불가."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None, "자료 부족"
    return a < b, f"{fmt.format(b)} → {fmt.format(a)}"


def bubble_checks(big4, flags, ctx, today):
    """해소 조건과 악화 신호. ctx 키: real_rate_20d(10년 실질금리 20거래일 변화 %p), hy_20d(하이일드 20거래일 변화 %p), hy_vs_low(하이일드 − 252거래일 최저 %p), support_ratio(최근 두 값 리스트), support_detail(C1 설명 문장),
    coverage(최근 두 AI 매출÷CAPEX 공개치 [(분기말, 값)]), anthropic_ratio(최근 세 컴퓨팅비용÷매출), crwv_cds([(일자, bp)] 오름차순),
    cloud_gain({티커: [(분기말, 클라우드 연환산 매출 증가÷1년 전 CAPEX)] 오름차순}), soxx_1m(SOXX 1개월 수익률), vix."""
    today = pd.Timestamp(today)
    h = big4.sort_values("period_end").dropna(subset=["capex_yoy"]).tail(2)
    cur = h.iloc[-1] if len(h) else None
    res, warn = [], []

    if len(h) == 2:
        now, before = h.capex_ocf.iloc[1], h.capex_ocf.iloc[0]
        ok = bool(now <= CAPEX_OCF_OK or before - now >= CAPEX_OCF_DROP_PP)
        d = f"{before:.0%} → {now:.0%} (75% 이하이거나 5%p 이상 하락이면 충족)"
    else:
        ok, d = None, "자료 부족"
    res.append(_item("A", "A1", "Big4 CAPEX ÷ 영업현금흐름이 75% 이하이거나 직전 분기보다 5%p 이상 낮아졌다", ok, d))
    if len(h) == 2:
        ok = h.fcf_ttm.iloc[1] > h.fcf_ttm.iloc[0]
        res.append(_item("A", "A2", "Big4 12개월 잉여현금흐름이 직전 분기보다 늘었다", ok, f"{h.fcf_ttm.iloc[0] / 1e8:,.0f}억 → {h.fcf_ttm.iloc[1] / 1e8:,.0f}억 달러"))
    else:
        res.append(_item("A", "A2", "Big4 12개월 잉여현금흐름이 직전 분기보다 늘었다", None, "자료 부족"))

    lv = lambda fid: flags.get(fid)  # noqa: E731
    res.append(_item("B", "B1", "소프트웨어 AI 매출화(F18)가 긍정이다", None if lv("F18") is None else lv("F18") == "긍정", f"F18 {lv('F18') or '없음'}"))
    cov = ctx.get("coverage") or []
    if len(cov) >= 2 and (today - pd.Timestamp(cov[-1][0])).days <= COVERAGE_MAX_AGE:
        ok, d = _less(cov[-2][1], cov[-1][1], "{:.2f}")   # 직전 < 최신이면 높아진 것
        res.append(_item("B", "B2", "AI 연환산 매출 ÷ CAPEX가 직전 공개치보다 높다", ok, d))
    else:
        last = cov[-1][0] if cov else "없음"
        res.append(_item("B", "B2", "AI 연환산 매출 ÷ CAPEX가 직전 공개치보다 높다", None,
                         f"최신 공개가 200일보다 오래됨(최신 {last}). 마이크로소프트가 2026-09 분기 개편 뒤 공개를 중단해 B5가 대신 본다"))
    cg = ctx.get("cloud_gain") or {}
    pairs = {t: [x for x in (cg.get(t) or []) if pd.notna(x[1])][-2:] for t in ("AMZN", "GOOGL")}
    if all(len(v) == 2 for v in pairs.values()):
        names = {"AMZN": "AWS", "GOOGL": "Google Cloud"}
        ok = all(v[1][1] > v[0][1] for v in pairs.values())
        d = " · ".join(f"{names[t]} {v[0][1]:.2f} → {v[1][1]:.2f}" for t, v in pairs.items())
        res.append(_item("B", "B5", "클라우드(AWS·Google Cloud) 연환산 매출 증가 ÷ 1년 전 CAPEX가 직전 분기보다 둘 다 높다", ok, d))
    else:
        res.append(_item("B", "B5", "클라우드(AWS·Google Cloud) 연환산 매출 증가 ÷ 1년 전 CAPEX가 직전 분기보다 둘 다 높다", None, "자료 부족"))
    ar = ctx.get("anthropic_ratio") or []
    if len(ar) >= 3:
        ok = bool(ar[-1] < ar[-2] < ar[-3])
        d = " → ".join(f"{x:.0%}" for x in ar[-3:]) + " (2분기 연속 하락이면 충족)"
    else:
        ok, d = None, "분기 3개 미만이라 판정 불가"
    res.append(_item("B", "B3", "Anthropic 매출 대비 컴퓨팅 비용 비율이 2분기 연속 낮아졌다", ok, d))
    res.append(_item("B", "B4", "AI 도입 확산(F17)이 긍정이다", None if lv("F17") is None else lv("F17") == "긍정", f"F17 {lv('F17') or '없음'}"))

    sr = ctx.get("support_ratio") or []
    ok, d = _less(sr[-1], sr[-2], "{:.0%}") if len(sr) >= 2 else (None, "자료 부족")
    if ctx.get("support_detail") and ok is not None:
        d = f"{d} · {ctx['support_detail']}"
    res.append(_item("C", "C1", "NVIDIA 고객 지원성 약정 ÷ 매출이 직전 분기보다 낮다", ok, d))
    res.append(_item("C", "C2", "NVIDIA 수요의 질(F16)이 정상이다", None if lv("F16") is None else lv("F16") == "정상", f"F16 {lv('F16') or '없음'}"))
    res.append(_item("C", "C3", "차입 의존 기업 런웨이(F11)가 경고가 아니다", None if lv("F11") is None else lv("F11") != "경고", f"F11 {lv('F11') or '없음'}"))
    cds = ctx.get("crwv_cds") or []
    if len(cds) >= 2 and (today - pd.Timestamp(cds[-1][0])).days <= CDS_MAX_AGE:
        ok, d = _less(cds[-1][1], cds[-2][1], "{:,.0f}bp")
        res.append(_item("C", "C4", "CoreWeave 5년 CDS가 직전 기록보다 낮다", ok, d))
    else:
        res.append(_item("C", "C4", "CoreWeave 5년 CDS가 직전 기록보다 낮다", None, f"최근 {CDS_MAX_AGE}일 안의 새 기록 없음(최신 {cds[-1][0] if cds else '없음'})"))
    rr = ctx.get("real_rate_20d")
    res.append(_item("C", "C5", "10년 실질금리가 20거래일 전보다 25bp 이상 낮다", None if rr is None else rr * 100 <= RATE_20D_BP,
                     "자료 부족" if rr is None else f"20거래일 변화 {rr * 100:+.0f}bp (−25bp 이하면 충족)"))
    hy = ctx.get("hy_20d")
    res.append(_item("C", "C6", "하이일드 스프레드가 20거래일 전보다 20bp 이상 낮다", None if hy is None else hy * 100 <= HY_20D_BP,
                     "자료 부족" if hy is None else f"20거래일 변화 {hy * 100:+.0f}bp (−20bp 이하면 충족)"))

    s1 = ctx.get("soxx_1m")
    res.append(_item("D", "D1", "SOXX 최근 1개월 수익률이 0 이상이다", None if s1 is None else s1 >= 0, "자료 부족" if s1 is None else f"{s1:+.1%}"))
    vix = ctx.get("vix")
    res.append(_item("D", "D2", "VIX가 25 미만이다", None if vix is None else vix < 25, "자료 부족" if vix is None else f"VIX {vix:.1f}"))

    bad = lambda fid, good: None if lv(fid) is None else lv(fid) not in good  # noqa: E731
    warn.append(_item("W", "W1", "GPU 임대가격(F15)이 주의 이상이다", None if lv("F15") is None else lv("F15") in {"주의", "경고"}, f"F15 {lv('F15') or '없음'}"))
    warn.append(_item("W", "W2", "수주잔고 선행(F4)이 긍정이 아니다", bad("F4", {"긍정"}), f"F4 {lv('F4') or '없음'}"))
    warn.append(_item("W", "W3", "클라우드 성장·수익성(F7)이 긍정이 아니다", bad("F7", {"긍정"}), f"F7 {lv('F7') or '없음'}"))
    warn.append(_item("W", "W4", "대만 AI 공급망(F8)이 강함이 아니다", bad("F8", {"강함"}), f"F8 {lv('F8') or '없음'}"))
    if len(h) == 2:
        drop = h.capex_yoy.iloc[0] - h.capex_yoy.iloc[1]
        f7_bad = bad("F7", {"긍정"})
        warn.append(_item("W", "W5", "CAPEX 증가율이 10%p 넘게 낮아졌는데 클라우드(F7)가 긍정이 아니다",
                          None if f7_bad is None else (drop >= CAPEX_DROP and f7_bad), f"CAPEX 증가율 변화 {-drop * 100:+.0f}%p, F7 {lv('F7') or '없음'}"))
    else:
        warn.append(_item("W", "W5", "CAPEX 증가율이 10%p 넘게 낮아졌는데 클라우드(F7)가 긍정이 아니다", None, "자료 부족"))
    bg = big4.sort_values("period_end").dropna(subset=["fcf_ttm"])
    if len(bg) >= 5 and 300 <= (bg.period_end.iloc[-1] - bg.period_end.iloc[-5]).days <= 430:
        a, b = bg.fcf_ttm.iloc[-5], bg.fcf_ttm.iloc[-1]
        warn.append(_item("W", "W6", "Big4 12개월 잉여현금흐름이 1년 전 같은 분기보다 줄었다", b < a, f"{a / 1e8:,.0f}억 → {b / 1e8:,.0f}억 달러"))
    else:
        warn.append(_item("W", "W6", "Big4 12개월 잉여현금흐름이 1년 전 같은 분기보다 줄었다", None, "자료 부족"))
    co = cur.capex_ocf if cur is not None and "capex_ocf" in h and pd.notna(cur.capex_ocf) else None
    warn.append(_item("W", "W7", "Big4 CAPEX ÷ 영업현금흐름이 100% 이상이다(번 돈보다 쓰는 돈이 크다)", None if co is None else bool(co >= W_CAPEX_OCF),
                      "자료 부족" if co is None else f"{co:.0%}"))
    hl = ctx.get("hy_vs_low")
    warn.append(_item("W", "W8", "하이일드 스프레드가 252거래일 최저보다 100bp 이상 높다", None if hl is None else hl * 100 >= W_HY_FROM_LOW_BP,
                      "자료 부족" if hl is None else f"최저 대비 {hl * 100:+.0f}bp"))
    if cds and (today - pd.Timestamp(cds[-1][0])).days <= CDS_MAX_AGE:
        warn.append(_item("W", "W9", "CoreWeave 5년 CDS가 1,000bp 이상이다", cds[-1][1] >= W_CDS_BP, f"{cds[-1][1]:,.0f}bp ({cds[-1][0]})"))
    else:
        warn.append(_item("W", "W9", "CoreWeave 5년 CDS가 1,000bp 이상이다", None, f"최근 {CDS_MAX_AGE}일 안의 새 기록 없음(최신 {cds[-1][0] if cds else '없음'})"))
    if len(ar) >= 3:
        warn.append(_item("W", "W10", "Anthropic 컴퓨팅 비용 ÷ 매출이 2분기 연속 올랐다", bool(ar[-1] > ar[-2] > ar[-3]), " → ".join(f"{x:.0%}" for x in ar[-3:])))
    else:
        warn.append(_item("W", "W10", "Anthropic 컴퓨팅 비용 ÷ 매출이 2분기 연속 올랐다", None, "분기 3개 미만이라 판정 불가"))
    return {"resolution": res, "warning": warn}


def bubble_changes(cur, prev):
    """지난 스냅샷 대비 예/아니오가 바뀐 점검 항목."""
    if not prev:
        return []
    before = {i["id"]: i["ok"] for i in prev.get("resolution", []) + prev.get("warning", [])}
    return [{"id": i["id"], "text": i["text"], "from": before[i["id"]], "to": i["ok"]}
            for i in cur["resolution"] + cur["warning"] if i["id"] in before and before[i["id"]] != i["ok"]]
