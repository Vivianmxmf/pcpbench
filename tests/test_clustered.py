"""Tests for the clustered class-conditional calibrator (Ding et al. 2023)."""

import numpy as np

from pcpbench.calibrators.clustered import ClusteredMondrian, get_quantile_threshold
from pcpbench.calibrators.quantile import split_quantile
from pcpbench.calibrators.split import SplitCP
from pcpbench.cli import main
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.runner import METHODS
from pcpbench.scores.softmax_scores import lac_score_matrix


def _true_scores(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    sm = lac_score_matrix(logits)
    return sm[np.arange(len(labels)), labels]


def _margin_logits(rng: np.random.Generator, counts: list[int], margins: list[float],
                   noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    K = len(counts)
    xs, ys = [], []
    for c, n in enumerate(counts):
        mu = np.zeros(K)
        mu[c] = margins[c]
        xs.append(mu + rng.normal(0.0, noise, size=(n, K)))
        ys.append(np.full(n, c))
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


def _random_logits(rng: np.random.Generator, counts: list[int]
                   ) -> tuple[np.ndarray, np.ndarray]:
    K = len(counts)
    y = np.concatenate([np.full(n, c) for c, n in enumerate(counts)]).astype(np.int64)
    x = rng.normal(0.0, 1.0, size=(len(y), K))
    x[np.arange(len(y)), y] += 1.0
    return x, y


def _loop_threshold(alpha: float) -> int:
    n = 1
    while np.ceil((n + 1) * (1 - alpha) / n) > 1:
        n += 1
    return n


def _rule(num_classes: int, n: int) -> tuple[int, int]:
    n_clustering = int(n * num_classes / (75 + num_classes))
    return n_clustering, int(np.floor(n_clustering / 2))


def test_registry_and_cli_run(capsys):
    assert "clustered" in METHODS
    assert METHODS["clustered"] is ClusteredMondrian
    rc = main(["run", "--dataset", "synthetic", "--method", "clustered",
               "--alpha", "0.1", "--seeds", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"n_classes"' in out and '"clustered"' in out


def test_quantile_threshold():
    assert get_quantile_threshold(0.1) == 9
    for alpha in (0.05, 0.2):
        assert get_quantile_threshold(alpha) == _loop_threshold(alpha)


def test_parameter_rule_uses_rarest_class_count():
    rng = np.random.default_rng(0)
    counts = [50] + [50 + 5 * c for c in range(1, 10)]
    x, y = _random_logits(rng, counts)
    fit = ClusteredMondrian().fit(x, y, alpha=0.1, seed=0)
    p = fit.resolved_params
    assert p["n_min"] == 50
    assert p["num_remaining_classes"] == 10
    assert (p["n_clustering"], p["num_clusters"]) == _rule(10, 50) == (5, 2)
    assert abs(p["frac_clustering"] - 5 / 50) < 1e-15

    counts = [50] * 60 + [70] * 40
    x, y = _random_logits(rng, counts)
    p = ClusteredMondrian().fit(x, y, alpha=0.1, seed=0).resolved_params
    assert (p["n_clustering"], p["num_clusters"]) == _rule(100, 50) == (28, 14)

    counts = [3, 5] + [40] * 8
    x, y = _random_logits(rng, counts)
    p = ClusteredMondrian().fit(x, y, alpha=0.1, seed=0).resolved_params
    assert p["n_thresh"] == 9 and p["n_min"] == 9
    assert p["num_remaining_classes"] == 8
    assert (p["n_clustering"], p["num_clusters"]) == _rule(8, 9)


def test_skip_rule_at_500_per_class_equals_split_on_calibration_part():
    K, n = 100, 500
    rng = np.random.default_rng(1)
    x, y = _random_logits(rng, [n] * K)
    fit = ClusteredMondrian().fit(x, y, alpha=0.1, seed=3)
    p = fit.resolved_params
    assert (p["n_clustering"], p["num_clusters"]) == _rule(K, n) == (285, 142)
    assert p["clustering_skipped"] is True
    assert fit.clustering_skipped is True
    assert np.all(fit.cluster_of_class == -1)
    cal = fit.calibration_mask
    assert 0 < cal.sum() < len(y)
    expected = split_quantile(_true_scores(x, y)[cal], 0.1)
    split_thr = float(SplitCP().fit(x[cal], y[cal], alpha=0.1).thresholds())
    assert abs(expected - split_thr) < 1e-12
    assert np.all(np.abs(fit.thresholds() - split_thr) < 1e-12)


def test_random_clustering_split_rate():
    K, n = 100, 500
    rng = np.random.default_rng(2)
    x, y = _random_logits(rng, [n] * K)
    fit = ClusteredMondrian().fit(x, y, alpha=0.1, seed=0)
    frac = fit.resolved_params["frac_clustering"]
    per_class = np.array([int((~fit.calibration_mask & (y == c)).sum()) for c in range(K)])
    assert len(np.unique(per_class)) > 1
    rate = float((~fit.calibration_mask).mean())
    sd = np.sqrt(frac * (1 - frac) / len(y))
    assert abs(rate - frac) < 5 * sd
    draw = np.random.default_rng(0).uniform(size=len(y)) < frac
    assert np.array_equal(~fit.calibration_mask, draw)


def test_rare_rule_eight_versus_nine_clustering_points():
    rng = np.random.default_rng(4)
    counts = [60, 60, 60, 8, 9, 60]
    x, y = _margin_logits(rng, counts, [0.5, 2.0, 3.5, 5.0, 6.5, 8.0])
    fit = ClusteredMondrian(n_clustering=9, num_clusters=2).fit(x, y, alpha=0.1, seed=0)
    assert fit.resolved_params["n_min"] == 9
    assert fit.resolved_params["frac_clustering"] == 1.0
    assert not fit.calibration_mask.any()
    assert fit.cluster_of_class[3] == -1
    assert fit.cluster_of_class[4] >= 0
    assert fit.n_rare == 1
    assert fit.clustering_skipped is False


def test_rare_rule_matches_recorded_mask():
    rng = np.random.default_rng(5)
    counts = [12, 14, 16, 18, 20, 40, 60, 80]
    x, y = _random_logits(rng, counts)
    fit = ClusteredMondrian(n_clustering=6, num_clusters=2).fit(x, y, alpha=0.1, seed=1)
    thresh = get_quantile_threshold(0.1)
    clustering = ~fit.calibration_mask
    rare = np.array([int((clustering & (y == c)).sum()) < thresh for c in range(len(counts))])
    assert fit.n_rare == int(rare.sum())
    assert 0 < fit.n_rare < len(counts)
    assert not fit.clustering_skipped
    assert np.all(fit.cluster_of_class[rare] == -1)
    assert np.all(fit.cluster_of_class[~rare] >= 0)
    null_thr = split_quantile(_true_scores(x, y)[fit.calibration_mask], 0.1)
    assert np.all(np.abs(fit.thresholds()[rare] - null_thr) < 1e-12)


def test_genuine_clustering_pairs_share_clusters():
    K, n = 4, 400
    rng = np.random.default_rng(6)
    x, y = _margin_logits(rng, [n] * K, [0.5, 0.5, 6.0, 6.0], noise=0.3)
    fit = ClusteredMondrian(n_clustering=100, num_clusters=2).fit(x, y, alpha=0.1, seed=0)
    coc = fit.cluster_of_class
    assert fit.clustering_skipped is False
    assert fit.n_rare == 0
    assert coc[0] == coc[1] and coc[2] == coc[3] and coc[0] != coc[2]
    assert abs(fit.resolved_params["frac_clustering"] - 100 / 400) < 1e-15
    cal = fit.calibration_mask
    s = _true_scores(x, y)
    for pair in ((0, 1), (2, 3)):
        expected = split_quantile(s[cal & np.isin(y, pair)], 0.1)
        for c in pair:
            assert abs(fit.thresholds()[c] - expected) < 1e-12


def test_reduces_to_split_with_no_clustering_points():
    ds = make_synthetic_shift(K=6, n_per_class=200, seed=1)
    xc, yc = ds.source_cal()
    xc = xc.astype(np.float64)
    fit = ClusteredMondrian(n_clustering=0).fit(xc, yc, alpha=0.1)
    assert fit.clustering_skipped is True
    assert fit.calibration_mask.all()
    assert fit.n_rare == 6
    split_thr = float(SplitCP().fit(xc, yc, alpha=0.1).thresholds())
    thr = fit.thresholds()
    assert thr.shape == (6,)
    assert np.all(np.abs(thr - split_thr) < 1e-12)


def test_determinism_under_seed():
    rng = np.random.default_rng(7)
    x, y = _random_logits(rng, [50] * 100)
    a = ClusteredMondrian().fit(x, y, alpha=0.1, seed=7)
    b = ClusteredMondrian().fit(x, y, alpha=0.1, seed=7)
    assert a.clustering_skipped is False
    assert np.array_equal(a.thresholds(), b.thresholds())
    assert np.array_equal(a.cluster_of_class, b.cluster_of_class)
    assert np.array_equal(a.calibration_mask, b.calibration_mask)
    c = ClusteredMondrian().fit(x, y, alpha=0.1, seed=8)
    assert not np.array_equal(a.calibration_mask, c.calibration_mask)


def test_shape_protocol_and_exchangeable_coverage():
    ds = make_synthetic_shift(seed=0)
    x, y = ds.source_cal()
    K = ds.n_classes
    rng = np.random.default_rng(11)
    first = np.zeros(len(y), dtype=bool)
    for c in range(K):
        idx = rng.permutation(np.where(y == c)[0])
        first[idx[: len(idx) // 2]] = True
    fit = ClusteredMondrian().fit(x[first], y[first], alpha=0.1, seed=0)
    assert fit.thresholds().shape == (K,)
    mask = fit.predict_set(x[~first])
    assert mask.dtype == bool and mask.shape == (int((~first).sum()), K)
    marginal = float(mask[np.arange(mask.shape[0]), y[~first]].mean())
    assert 0.85 <= marginal <= 0.95


def test_default_rule_on_synthetic_skips_clustering():
    K, n_per_class = 12, 500
    ds = make_synthetic_shift(K=K, n_per_class=n_per_class, seed=0)
    xc, yc = ds.source_cal()
    n_min = max(int(np.bincount(yc, minlength=K).min()), get_quantile_threshold(0.1))
    fit = ClusteredMondrian().fit(xc, yc, alpha=0.1, seed=0)
    p = fit.resolved_params
    assert p["n_min"] == n_min
    assert (p["n_clustering"], p["num_clusters"]) == _rule(K, n_min)
    assert p["num_clusters"] >= K
    assert p["clustering_skipped"] is True
    null_thr = split_quantile(_true_scores(xc, yc)[fit.calibration_mask], 0.1)
    assert np.all(np.abs(fit.thresholds() - null_thr) < 1e-12)


def test_default_rule_at_50_per_class_k10_skips_in_every_seed():
    """The controlled sweep's K 10 cell has 50 calibration points per class."""
    K, n = 10, 50
    n_thresh = get_quantile_threshold(0.1)
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        x, y = _random_logits(rng, [n] * K)
        fit = ClusteredMondrian().fit(x, y, alpha=0.1, seed=seed)
        p = fit.resolved_params
        assert (p["n_clustering"], p["num_clusters"]) == _rule(K, n) == (5, 2)
        assert fit.n_rare >= K - p["num_clusters"]
        assert fit.clustering_skipped is True
        assert np.all(fit.cluster_of_class == -1)
        cal = fit.calibration_mask
        expected = split_quantile(_true_scores(x, y)[cal], 0.1)
        assert np.all(np.abs(fit.thresholds() - expected) < 1e-12)
        assert n_thresh == 9
