import numpy as np
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.calibrators.split import SplitCP
from pcpbench.calibrators.mondrian import SourceMondrian


def _worst_class(mask, y, K):
    cov = []
    for c in range(K):
        m = y == c
        if m.any():
            cov.append(mask[np.where(m)[0], c].mean())
    return min(cov)


def _make_heterogeneous_logits(rng, K, n_per_class, margins, noise_std):
    logits, labels = [], []
    for c in range(K):
        mu = np.zeros(K)
        mu[c] = margins[c]
        x = mu + rng.normal(0.0, noise_std, size=(n_per_class, K))
        logits.append(x)
        labels.append(np.full(n_per_class, c))
    return (np.concatenate(logits).astype(np.float32),
            np.concatenate(labels).astype(np.int64))


def test_mondrian_improves_worst_class_with_heterogeneous_difficulty():
    K = 8
    n_per_class = 500
    margins = np.array([4.0 - 0.4 * c for c in range(K)])
    rng = np.random.default_rng(0)
    xc, yc = _make_heterogeneous_logits(rng, K, n_per_class, margins, noise_std=1.0)
    xt, yt = _make_heterogeneous_logits(rng, K, n_per_class, margins, noise_std=1.0)

    split_mask = SplitCP().fit(xc, yc, alpha=0.1).predict_set(xt)
    mond_mask = SourceMondrian().fit(xc, yc, alpha=0.1).predict_set(xt)

    split_worst = _worst_class(split_mask, yt, K)
    mond_worst = _worst_class(mond_mask, yt, K)

    assert split_worst < 0.85
    assert mond_worst >= split_worst + 0.05
    assert mond_worst >= 0.85


def test_mondrian_thresholds_are_per_class():
    ds = make_synthetic_shift(K=5, n_per_class=200, seed=0)
    xc, yc = ds.source_cal()
    fit = SourceMondrian().fit(xc, yc, alpha=0.1)
    assert fit.thresholds().shape == (5,)
