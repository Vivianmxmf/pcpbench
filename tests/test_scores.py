import numpy as np
from pcpbench.scores.softmax_scores import softmax, lac_score_matrix, lac_scores


def test_softmax_rows_sum_to_one():
    z = np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 0.0]])
    p = softmax(z)
    assert np.allclose(p.sum(1), 1.0)


def test_lac_matrix_matches_true_label_scores():
    z = np.random.default_rng(0).normal(size=(20, 4))
    y = np.random.default_rng(1).integers(0, 4, size=20)
    m = lac_score_matrix(z)
    s_true = lac_scores(softmax(z), y)
    assert np.allclose(m[np.arange(20), y], s_true)
    assert (m >= 0).all() and (m <= 1).all()


import pytest

from pcpbench.scores.softmax_scores import aps_scores


def test_score_matrix_aps_matches_non_randomized_aps_scores():
    from pcpbench.scores.matrix import score_matrix
    rng = np.random.default_rng(3)
    for K in (2, 5, 12, 40):
        z = rng.normal(0.0, 2.0, size=(200, K))
        sm = score_matrix(z, "aps")
        assert sm.shape == (200, K)
        p = softmax(z)
        for c in range(K):
            ref = aps_scores(p, np.full(200, c), randomized=False)
            assert np.max(np.abs(sm[:, c] - ref)) <= 1e-12


def test_score_matrix_aps_true_label_matches_aps_scores():
    from pcpbench.scores.matrix import score_matrix
    rng = np.random.default_rng(4)
    z = rng.normal(size=(300, 7))
    y = rng.integers(0, 7, size=300)
    sm = score_matrix(z, "aps")
    ref = aps_scores(softmax(z), y, randomized=False)
    assert np.max(np.abs(sm[np.arange(300), y] - ref)) <= 1e-12
    assert np.allclose(sm.min(axis=1), softmax(z).max(axis=1))
    assert (sm > 0).all() and (sm <= 1.0 + 1e-12).all()


def test_score_matrix_aps_k4_all_tied_matches_aps_scores():
    from pcpbench.scores.matrix import score_matrix
    z = np.zeros((1, 4))
    sm = score_matrix(z, "aps")
    assert np.allclose(sm[0], [0.25, 0.5, 0.75, 1.0])
    p = softmax(z)
    for c in range(4):
        assert np.array_equal(sm[:, c], aps_scores(p, np.full(1, c), randomized=False))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("K", (4, 17, 50, 100))
def test_score_matrix_aps_equals_aps_scores_under_exact_ties(K, dtype):
    """Exact probability ties (copied logit columns) at K of 17 or more, where numpy's default
    sort is no longer stable: every column must equal ``aps_scores`` byte for byte, because both
    sort with the same default ``np.argsort(-p, axis=1)`` on the same float64 probabilities."""
    from pcpbench.scores.matrix import score_matrix
    rng = np.random.default_rng(1000 + K)
    n = 150
    z = rng.normal(0.0, 2.0, size=(n, K)).astype(dtype)
    z[:, 1] = z[:, 0]
    z[:50, K - 1] = z[:50, K // 2]
    sm = score_matrix(z, "aps")
    p = softmax(np.asarray(z, dtype=np.float64))
    worst = 0.0
    for c in range(K):
        ref = aps_scores(p, np.full(n, c), randomized=False)
        worst = max(worst, float(np.max(np.abs(sm[:, c] - ref))))
    assert worst == 0.0


def test_score_matrix_lac_equals_lac_score_matrix():
    from pcpbench.scores.matrix import score_matrix
    z = np.random.default_rng(5).normal(size=(50, 6))
    assert np.array_equal(score_matrix(z, "lac"), lac_score_matrix(z))
    assert np.array_equal(score_matrix(z), lac_score_matrix(z))


def test_score_matrix_casts_float32_logits_to_float64():
    from pcpbench.scores.matrix import score_matrix
    z = np.random.default_rng(6).normal(size=(40, 5)).astype(np.float32)
    for name in ("lac", "aps"):
        out = score_matrix(z, name)
        assert out.dtype == np.float64
        assert np.array_equal(out, score_matrix(z.astype(np.float64), name))


def test_score_matrix_unknown_score_raises():
    from pcpbench.scores.matrix import SCORES, score_matrix
    assert SCORES == ("lac", "aps")
    with pytest.raises(ValueError):
        score_matrix(np.zeros((2, 3)), "raps")
