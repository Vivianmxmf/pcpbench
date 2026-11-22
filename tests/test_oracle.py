import numpy as np
import pytest
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.calibrators.mondrian import SourceMondrian
from pcpbench.calibrators.oracle import OracleTargetMondrian


def _worst(mask, y, K):
    return min(mask[np.where(y == c)[0], c].mean() for c in range(K) if (y == c).any())


def test_oracle_upper_bounds_source_mondrian():
    ds = make_synthetic_shift(K=8, n_per_class=500, shift=0.7, seed=0)
    xc, yc = ds.source_cal(); xt, yt = ds.target_test()
    src = SourceMondrian().fit(xc, yc, alpha=0.1).predict_set(xt)
    orc = OracleTargetMondrian().fit(xc, yc, target_labeled=(xt, yt),
                                     alpha=0.1).predict_set(xt)
    assert _worst(orc, yt, 8) >= _worst(src, yt, 8) - 0.02


def test_oracle_requires_target_labeled():
    ds = make_synthetic_shift(K=8, n_per_class=500, shift=0.7, seed=0)
    xc, yc = ds.source_cal()
    with pytest.raises(ValueError):
        OracleTargetMondrian().fit(xc, yc, target_labeled=None, alpha=0.1)
