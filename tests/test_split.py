import numpy as np
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.calibrators.split import SplitCP


def test_splitcp_marginal_coverage_in_distribution():
    ds = make_synthetic_shift(K=6, n_per_class=400, shift=0.0, seed=0)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    fit = SplitCP().fit(xc, yc, alpha=0.1)
    mask = fit.predict_set(xt)
    cov = mask[np.arange(len(yt)), yt].mean()
    assert 0.86 <= cov <= 0.94
