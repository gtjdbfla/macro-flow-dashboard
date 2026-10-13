import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.stats import cond_stats, episodes, split_valid  # noqa: E402


def test_episodes_split_by_gap():
    assert episodes(np.array([1, 2, 3, 30, 31, 90]), 5).tolist() == [0, 0, 0, 1, 1, 2]


def test_purge_removes_windows_crossing_split_end():
    idx = pd.bdate_range("2018-12-20", "2019-01-10")
    ok = split_valid(idx, "train", horizon_rows=3)
    last_ok = idx[ok][-1]
    assert last_ok <= pd.Timestamp("2018-12-31") and (idx[idx <= "2018-12-31"][-3] >= last_ok)


def test_cond_stats_detects_planted_effect_and_ignores_noise():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(0, 0.01, n)
    mask = np.zeros(n, bool)
    mask[rng.choice(n, 400, replace=False)] = True
    noise = cond_stats(y, np.ones(n, bool), mask, 1)
    assert noise["ci_lo"] < 0 < noise["ci_hi"]  # 효과 없음 -> CI가 0 포함
    y2 = y + mask * 0.01
    eff = cond_stats(y2, np.ones(n, bool), mask, 1)
    assert eff["ci_lo"] > 0  # 심은 효과 검출


def test_overlap_shrinks_effective_n():
    n = 2000
    y = np.random.default_rng(1).normal(0, 0.01, n)
    mask = np.zeros(n, bool)
    mask[100:160] = True  # 연속 60일 = 한 episode
    s = cond_stats(y, np.ones(n, bool), mask, 20)
    assert s["n"] == 60 and s["n_eff"] == 1
