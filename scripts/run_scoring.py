"""점수·한국 전달 모델·섹터 지도·유사국면 계산 -> data/results/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest import analog  # noqa: E402
from scoring import briefing, engine, korea_model, sector_score  # noqa: E402
from src.data import DATA_DIR  # noqa: E402

out = DATA_DIR / "results"
out.mkdir(exist_ok=True)

M, C, V, S, flags, rep = engine.run()
V.to_parquet(out / "score_validation.parquet")
hist, kor = korea_model.run()
S = S.join(hist["korea_score"])
S.to_parquet(out / "scores_history.parquet")
hist.to_parquet(out / "korea_model_history.parquet")
comp_now = C.dropna(how="all").iloc[-1]
sec, ind, asof = sector_score.build()
sec.to_parquet(out / "sector_snapshot.parquet")
ind.to_parquet(out / "industry_snapshot.parquet")
meta, nb, fwd = analog.run()
nb.to_parquet(out / "analog_neighbors.parquet")
fwd.to_parquet(out / "analog_forward.parquet")

last = S.dropna(subset=["macro"]).index[-1]
summary = {
    "as_of": str(last.date()),
    "scores": {k: (None if S[k].loc[last] != S[k].loc[last] else float(S[k].loc[last])) for k in S.columns},
    "validated": flags, "oos_ic": rep, "korea_model": kor,
    "components_now": {f"{a}|{b}": (None if v != v else float(v)) for (a, b), v in comp_now.items()},
    "analog": meta,
}
summary["briefing"] = briefing.build(summary)
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("as_of", summary["as_of"], {k: None if v is None else round(v) for k, v in summary["scores"].items()})
print("validated", flags)
print("OOS IC(test)", {k: v["test"]["ic"] for k, v in rep.items()})
print("korea eval", kor["eval"])
print("analog", {k: meta[k] for k in ("n_days", "n_episodes", "nearest", "threshold")})
inc = V[V.included][["score", "component", "ic_train", "ic_val", "ic_test", "weight_norm"]]
print(inc.round(3).to_string() if len(inc) else "채택된 구성요소 없음")
