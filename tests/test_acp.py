import numpy as np
import pytest

from pcpbench.calibrators.acp import ACPCalibrator, _audit_probs
from pcpbench.datasets.synthetic import make_synthetic_shift


def _sample_labeled_per_class(logits, labels, m, K, rng):
    parts_x, parts_y = [], []
    for c in range(K):
        idx_c = np.where(labels == c)[0]
        take = min(m, len(idx_c))
        chosen = rng.choice(idx_c, size=take, replace=False)
        parts_x.append(logits[chosen])
        parts_y.append(labels[chosen])
    return np.concatenate(parts_x), np.concatenate(parts_y)


def test_acp_requires_target_labeled():
    ds = make_synthetic_shift(K=6, n_per_class=200, shift=0.5, seed=0)
    xc, yc = ds.source_cal()
    with pytest.raises(ValueError):
        ACPCalibrator().fit(xc, yc, target_labeled=None, alpha=0.1)


def test_acp_predict_set_shape_and_valid():
    ds = make_synthetic_shift(K=6, n_per_class=300, shift=0.5, seed=0)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    K = ds.n_classes

    rng = np.random.default_rng(1)
    audit_x, audit_y = _sample_labeled_per_class(xt, yt, m=30, K=K, rng=rng)

    fitted = ACPCalibrator().fit(
        xc, yc, target_labeled=(audit_x, audit_y), alpha=0.1, seed=0
    )
    mask = fitted.predict_set(xt)

    assert mask.shape == (len(yt), K)
    assert mask.dtype == np.bool_
    assert np.all(mask.sum(axis=1) >= 1)


def test_acp_audited_probs_sum_to_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 5))
    for r in (0.0, 0.3, 0.5, 0.7, 1.0):
        r_hat = np.full(20, r)
        m = _audit_probs(logits, r_hat)
        assert m.shape == (20, 5)
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-9)


def test_acp_constant_audit_labels_no_crash():
    ds = make_synthetic_shift(K=6, n_per_class=300, shift=0.5, seed=0)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    K = ds.n_classes

    yhat = np.argmax(xt, axis=1)
    correct_idx = np.where(yhat == yt)[0]
    assert len(correct_idx) >= 60, "scenario needs enough correct target points"
    rng = np.random.default_rng(2)
    chosen = rng.choice(correct_idx, size=60, replace=False)
    audit_x, audit_y = xt[chosen], yt[chosen]
    assert np.all(np.argmax(audit_x, axis=1) == audit_y)

    fitted = ACPCalibrator().fit(
        xc, yc, target_labeled=(audit_x, audit_y), alpha=0.1, seed=0
    )
    mask = fitted.predict_set(xt)

    assert mask.shape == (len(yt), K)
    assert mask.dtype == np.bool_
    assert np.all(mask.sum(axis=1) >= 1)


def test_acp_marginal_coverage_reasonable():
    ds = make_synthetic_shift(K=6, n_per_class=500, shift=0.3, seed=0)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    K = ds.n_classes
    alpha = 0.1

    rng = np.random.default_rng(3)
    audit_x, audit_y = _sample_labeled_per_class(xt, yt, m=40, K=K, rng=rng)

    fitted = ACPCalibrator().fit(
        xc, yc, target_labeled=(audit_x, audit_y), alpha=alpha, seed=0
    )
    mask = fitted.predict_set(xt)
    cov = mask[np.arange(len(yt)), yt].mean()

    assert cov >= 1 - alpha - 0.15


def test_acp_is_deterministic():
    ds = make_synthetic_shift(K=6, n_per_class=300, shift=0.5, seed=0)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    K = ds.n_classes

    rng = np.random.default_rng(5)
    audit_x, audit_y = _sample_labeled_per_class(xt, yt, m=25, K=K, rng=rng)

    fit_a = ACPCalibrator().fit(
        xc, yc, target_labeled=(audit_x, audit_y), alpha=0.1, seed=7
    )
    fit_b = ACPCalibrator().fit(
        xc, yc, target_labeled=(audit_x, audit_y), alpha=0.1, seed=7
    )

    assert np.array_equal(fit_a.thresholds(), fit_b.thresholds())
    assert np.array_equal(fit_a.predict_set(xt), fit_b.predict_set(xt))
