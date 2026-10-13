"""Look-ahead 검증: 시점 T까지의 정보만으로 만든 feature가, 전체 데이터로 만든 feature의 T 이전 행과 같아야 한다.
다르면 미래 정보가 과거 행에 새어든 것이다.  실행: pytest tests -q  (data/parquet 필요)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build import build_all  # noqa: E402

CUTS = ["2012-06-15", "2020-03-20", "2024-09-30"]


@pytest.fixture(scope="module")
def full():
    return build_all()


@pytest.mark.parametrize("cut", CUTS)
def test_truncation_invariance(full, cut):
    part = build_all(asof=cut)
    bad = []
    for layer, df_part in part.items():
        df_full = full[layer].loc[df_part.index]
        for c in df_part.columns:
            a, b = df_part[c], df_full[c]
            if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
                same = (a.astype(object).where(a.notna(), "NA") == b.astype(object).where(b.notna(), "NA")).all()
            else:
                m = a.notna() | b.notna()
                same = np.allclose(a[m].to_numpy(float), b[m].to_numpy(float), rtol=1e-6, atol=1e-9,
                                   equal_nan=True)
            if not same:
                bad.append(f"{layer}.{c}")
    assert not bad, f"미래 정보 누수 의심 {len(bad)}개: {bad[:15]}"
