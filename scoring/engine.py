"""Phase 4 — 점수화. 원칙: 가중치는 과거 예측력으로 정한다. 임의 가중치 금지.

각 점수 = 구성요소(0~1, '위험자산에 우호적일수록 큼'으로 방향을 사전 고정)의 가중평균 x 100.
  - 구성요소 방향은 코드에 사전 고정(가설). 데이터를 본 뒤 부호를 뒤집지 않는다.
  - 가중치: train+val(2005~2022)에서 구성요소 vs SPY 20일 선행수익의 비겹침 Spearman IC를 구해,
    train·val 모두 가설 방향(IC>0)이고 풀링 t ≥ 2인 것만 채택, 가중치 ∝ 풀링 IC.
  - test(2023~)는 가중치에 쓰지 않고 합성 점수의 표본 외 IC만 보고한다.
  - 채택된 구성요소가 하나도 없으면 점수는 '미검증'으로 표시하고 동일가중 기술적 요약으로만 쓴다.
구성요소는 expanding 순위(과거만)이므로 점수 자체에 look-ahead는 없다. 가중치 선택은 train+val 정보를 쓴다."""
import numpy as np
import pandas as pd

from backtest.forward_returns import Fwd
from backtest.stats import split_valid
from src.data import PARQUET
from src.features.common import Ctx, exp_rank

H = 20
# (표시명, 컬럼, 방향 +1=클수록 우호 / -1=작을수록 우호, 이미 rank인지)
COMPONENTS = {
    "macro": [
        ("성장 가속", "state__growth_z", +1, False), ("물가 둔화", "state__inflation_z", -1, False),
        ("정책금리 6M 변화(인하 우호)", "state__policy_chg_6m", -1, False),
        ("수익률곡선 10s2s", "macro__Curve10s2s__level_rank", +1, True),
        ("신규실업수당 4주평균(낮을수록 우호)", "macro__Claims__ma4w_rank", -1, True),
        ("달러 60일 변화(약세 우호)", "macro__DXY__chg_60d_rank", -1, True),
    ],
    "liquidity": [
        ("순유동성 60일 변화", "liq__NetLiquidity__chg_60d_rank", +1, True),
        ("금융여건 NFCI(낮을수록 우호)", "liq__NFCI__level_rank", -1, True),
        ("광의달러 60일 변화(약세 우호)", "liq__BroadUSD__chg_60d_rank", -1, True),
    ],
    "risk": [
        ("VIX 수준(낮을수록 우호)", "risk__VIX__level_rank", -1, True),
        ("MOVE 수준(낮을수록 우호)", "risk__MOVE__level_rank", -1, True),
        ("HYG/LQD 20일 변화", "risk__HYG_LQD__chg_20d_rank", +1, True),
        ("SPY 고점 대비 위치", "risk__SPY__drawdown", +1, False),
        ("섹터 200일선 상회 비율", "sector__breadth__sectors_above_200dma", +1, False),
    ],
    "flow": [
        ("S&P500 선물 투기 순포지션", "cftc__SP500_E-mini__levmoney_net_rank", +1, True),
        ("S&P500 선물 자산운용 순포지션", "cftc__SP500_E-mini__assetmgr_net_rank", +1, True),
        ("나스닥100 선물 투기 순포지션", "cftc__Nasdaq100_E-mini__levmoney_net_rank", +1, True),
        ("나스닥100 선물 자산운용 순포지션", "cftc__Nasdaq100_E-mini__assetmgr_net_rank", +1, True),
    ],
}
TITLES = {"macro": "Macro", "liquidity": "Liquidity", "risk": "Risk Appetite", "flow": "Institutional Flow"}


def load_frame() -> pd.DataFrame:
    parts = []
    for n in ("macro", "liquidity", "risk", "sector", "flow"):
        parts.append(pd.read_parquet(PARQUET / f"features_{n}.parquet"))
    parts.append(pd.read_parquet(PARQUET / "features_state.parquet").add_prefix("state__"))
    return pd.concat(parts, axis=1)


def component_frame(M: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for score, comps in COMPONENTS.items():
        for label, col, sign, is_rank in comps:
            if col not in M:
                continue
            s = M[col].astype(float)
            r = s if is_rank else exp_rank(s, 252)
            out[(score, label)] = r if sign > 0 else 1 - r
    return pd.DataFrame(out)


def _ic(x, y, mask):
    d = pd.concat([x[mask], y[mask]], axis=1).dropna()
    if len(d) < 20:
        return np.nan, len(d)
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()), len(d)


def validate(C: pd.DataFrame, idx, y: np.ndarray) -> pd.DataFrame:
    ys = pd.Series(y, index=idx)
    samp = np.arange(len(idx)) % H == 0
    sp = {k: split_valid(idx, k, H + 1) & samp for k in ("train", "val", "test")}
    sp["pool"] = sp["train"] | sp["val"]
    rows = []
    for (score, label) in C.columns:
        x = C[(score, label)]
        r = {"score": score, "component": label}
        for k, m in sp.items():
            ic, n = _ic(x, ys, m)
            r[f"ic_{k}"], r[f"n_{k}"] = ic, n
        r["t_pool"] = r["ic_pool"] * np.sqrt(max(r["n_pool"] - 2, 1) / max(1e-9, 1 - r["ic_pool"] ** 2)) \
            if np.isfinite(r["ic_pool"]) else np.nan
        r["included"] = bool(r["ic_train"] > 0 and r["ic_val"] > 0 and r["t_pool"] >= 2)
        r["weight"] = r["ic_pool"] if r["included"] else 0.0
        rows.append(r)
    v = pd.DataFrame(rows)
    for s, g in v.groupby("score"):
        tot = g["weight"].sum()
        v.loc[g.index, "weight_norm"] = g["weight"] / tot if tot > 0 else 0.0
    return v


def compose(C: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for score in COMPONENTS:
        cols = [c for c in C.columns if c[0] == score]
        sub = C[cols]
        w = V[V.score == score].set_index("component")["weight_norm"].reindex([c[1] for c in cols]).fillna(0).values
        validated = w.sum() > 0
        if not validated:
            w = np.ones(len(cols))
        W = pd.DataFrame(np.tile(w, (len(sub), 1)), index=sub.index, columns=cols) * sub.notna()
        out[score] = 100 * (sub.fillna(0) * W).sum(axis=1) / W.sum(axis=1).replace(0, np.nan)
        out[f"{score}_validated"] = validated
    return pd.DataFrame({k: v for k, v in out.items() if not k.endswith("_validated")}), \
        {k: bool(v) for k, v in out.items() if k.endswith("_validated")}


def run():
    ctx = Ctx()
    fw = Fwd(ctx)
    M = load_frame().reindex(ctx.cal)
    C = component_frame(M)
    y, _ = fw.abs("SPY", H)
    V = validate(C, ctx.cal, y)
    S, flags = compose(C, V)
    # 합성 점수의 분할별 표본 외 IC
    ys = pd.Series(y, index=ctx.cal)
    samp = np.arange(len(ctx.cal)) % H == 0
    rep = {}
    for score in S.columns:
        rep[score] = {}
        for k in ("train", "val", "test"):
            ic, n = _ic(S[score], ys, split_valid(ctx.cal, k, H + 1) & samp)
            rep[score][k] = {"ic": None if not np.isfinite(ic) else float(ic), "n": n}
    return M, C, V, S, flags, rep
