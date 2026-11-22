import numpy as np
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.label_complexity import label_complexity_curve
from pcpbench.metrics.oracle_ceiling import oracle_ceiling


def test_label_complexity_curve_improves_with_labels():
    ds = make_synthetic_shift(K=8, n_per_class=500, shift=0.7, seed=0)
    curve = label_complexity_curve(ds, alpha=0.1, m_grid=[2, 10, 50], seed=0)
    assert curve[50]["worst_class"] >= curve[10]["worst_class"] - 0.02


def test_oracle_ceiling_is_bounded_and_finite():
    ds = make_synthetic_shift(K=6, n_per_class=400, shift=0.6, seed=0)
    xt, yt = ds.target_test()
    gains = oracle_ceiling(xt, yt, alpha=0.1)
    assert np.all(np.isfinite(gains[~np.isnan(gains)]))
    assert np.nanmin(gains) >= 1.0 - 1e-6
