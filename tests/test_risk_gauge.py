"""위험 온도계: 확장 백분위가 미래 값에 영향받지 않아야 한다."""
import numpy as np
import pandas as pd

from scoring.risk_gauge import exprank, level_of


def test_exprank_ignores_future():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=1600))
    a = exprank(s, minn=300)
    b = exprank(s.iloc[:1200], minn=300)
    pd.testing.assert_series_equal(a.iloc[:1200], b)


def test_level_boundaries():
    assert [level_of(x) for x in (0.0, 0.39, 0.4, 0.6, 0.79, 0.8, 1.0)] == ["낮음", "낮음", "보통", "높음", "높음", "매우 높음", "매우 높음"]
