"""Tests for the validity floor of worst-class coverage."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pcpbench.metrics.validity_floor import (
    FloorEstimate,
    coverage_law,
    exact_min_coverage,
    expected_min_coverage,
    frac_significantly_undercovered,
    seed_averaged_floor,
    significance_cutoff,
    undercovered_false_positive_rate,
)


def test_coverage_law() -> None:
    from pcpbench.calibrators.quantile import split_quantile

    for n in (8, 9, 10, 19, 25, 29, 50, 137):
        a, b = coverage_law(n, 0.1)
        q = split_quantile(np.arange(1, n + 1, dtype=float), 0.1)
        assert a == (n + 1 if np.isinf(q) else int(q))
        assert a + b == n + 1
    assert coverage_law(50, 0.1) == (46, 5)
    assert coverage_law(9, 0.1) == (9, 1)
    assert coverage_law(8, 0.1) == (9, 0)


def test_rank_matches_closed_form_everywhere() -> None:
    """With the quantile boundary fix, the rank read from split_quantile equals the exact rule l
    = floor((n + 1) * alpha) at every n_cal in 1..1000 at alpha 0.1, computed here in integer
    arithmetic."""
    mismatches = [n for n in range(1, 1001) if coverage_law(n, 0.1)[1] != (n + 1) // 10]
    assert mismatches == []


def test_single_class_matches_beta_binomial_mean() -> None:
    a, b = coverage_law(50, 0.1)
    est = expected_min_coverage(1, 50, 50)
    assert isinstance(est, FloorEstimate)
    assert abs(est.mean - a / (a + b)) < 0.005


def test_monotone_in_k() -> None:
    means = [expected_min_coverage(k, 50, 50).mean for k in (10, 25, 50, 100)]
    assert all(x > y for x, y in zip(means, means[1:]))


def test_floor_brackets() -> None:
    """The brackets below come from an independent simulation written apart from this module."""
    assert 0.79 <= expected_min_coverage(10, 50, 50).mean <= 0.81
    assert 0.705 <= expected_min_coverage(100, 50, 50).mean <= 0.73
    assert 0.545 <= expected_min_coverage(1000, 25, 25).mean <= 0.575


def test_false_positive_rate_exact() -> None:
    n_cal, n_eval, level = 50, 50, 0.9
    a, b = coverage_law(n_cal, 0.1)
    c = int(np.ceil(level * n_eval - 1e-12)) - 1
    expected = float(stats.betabinom.cdf(c, n_eval, a, b))
    got = undercovered_false_positive_rate(n_cal, n_eval)
    assert abs(got - expected) < 1e-12
    assert 0.36 <= got <= 0.38
    a9, b9 = coverage_law(9, 0.1)
    assert b9 == 1
    exp9 = float(stats.betabinom.cdf(c, n_eval, a9, b9))
    assert exp9 > 0.0
    assert abs(undercovered_false_positive_rate(9, n_eval) - exp9) < 1e-12
    assert undercovered_false_positive_rate(8, 50) == 0.0


NULL_PAIRS = ((50, 50), (25, 25), (500, 500), (50, 10))


def _brute_cutoff(n_eval: int, n_cal: int, alpha: float, q: float) -> int:
    a, b = coverage_law(n_cal, alpha)
    if b == 0:
        return n_eval - 1
    best = -1
    for c in range(n_eval + 1):
        if stats.betabinom.cdf(c, n_eval, a, b) <= q:
            best = c
    return best


@pytest.mark.parametrize("n_eval,n_cal", NULL_PAIRS)
def test_cutoff_is_exact_and_maximal(n_eval: int, n_cal: int) -> None:
    q = 0.05
    a, b = coverage_law(n_cal, 0.1)
    c = significance_cutoff(n_eval, n_cal, 0.1, q)
    assert c == _brute_cutoff(n_eval, n_cal, 0.1, q)
    assert float(stats.betabinom.cdf(c, n_eval, a, b)) <= q
    assert float(stats.betabinom.cdf(c + 1, n_eval, a, b)) > q


def test_cutoff_when_coverage_is_one() -> None:
    assert coverage_law(8, 0.1)[1] == 0
    assert significance_cutoff(50, 8, 0.1, 0.05) == 49
    assert coverage_law(9, 0.1)[1] == 1
    assert significance_cutoff(50, 9, 0.1, 0.05) == _brute_cutoff(50, 9, 0.1, 0.05) < 49


def _simulate_flagged(
    coverage: np.ndarray | None, n_cal: int, n_eval: int, K: int, seeds: int, q: float
) -> float:
    a, b = coverage_law(n_cal, 0.1)
    rates = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        p = rng.beta(a, b, size=K) if coverage is None else coverage
        labels = np.repeat(np.arange(K), n_eval)
        covered = rng.random(K * n_eval) < p[labels]
        mask = np.zeros((K * n_eval, K), dtype=bool)
        mask[np.arange(K * n_eval), labels] = covered
        n_cal_arr = np.full(K, n_cal, dtype=np.int64)
        rates.append(frac_significantly_undercovered(mask, labels, K, n_cal_arr, 0.1, q))
    return float(np.mean(rates))


def test_significance_metric_null_rate() -> None:
    """Under exact validity (Beta coverages from the calibration law, Binomial evaluation counts)
    the flagged fraction sits at or just below q."""
    q = 0.05
    rate = _simulate_flagged(None, 50, 50, 200, 20, q)
    assert q - 0.02 <= rate <= q + 0.01


def test_significance_metric_power() -> None:
    """Power at true per-class coverage 0.80 with 50 evaluation points."""
    q, K = 0.05, 200
    cov = np.full(K, 0.80)
    c50 = significance_cutoff(50, 50, 0.1, q)
    exact50 = float(stats.binom.cdf(c50, 50, 0.80))
    assert abs(_simulate_flagged(cov, 50, 50, K, 20, q) - exact50) < 0.02
    assert _simulate_flagged(cov, 500, 50, K, 20, q) > 0.5


def test_significance_metric_excludes_untestable_classes() -> None:
    labels = np.array([0, 0, 1, 1])
    mask = np.zeros((4, 3), dtype=bool)
    frac = frac_significantly_undercovered(mask, labels, 3, np.array([50, 0, 50]))
    assert frac == 1.0
    assert np.isnan(frac_significantly_undercovered(mask, labels, 3, np.array([0, 0, 50])))


def test_significance_metric_nan_when_empty() -> None:
    mask = np.zeros((0, 3), dtype=bool)
    labels = np.zeros(0, dtype=int)
    assert np.isnan(frac_significantly_undercovered(mask, labels, 3, np.array([50, 50, 50])))


def test_per_class_arrays() -> None:
    cal = np.array([71, 150, 73, 74, 75])
    ev = np.array([70, 150, 72, 74, 74])
    est = expected_min_coverage(5, cal, ev)
    assert est.n_cal is None and est.n_eval is None
    lo = expected_min_coverage(5, int(cal.min()), int(ev.min())).mean
    hi = expected_min_coverage(5, int(cal.max()), int(ev.max())).mean
    assert min(lo, hi) <= est.mean <= max(lo, hi)


def test_seed_averaging_raises_floor() -> None:
    avg = seed_averaged_floor(100, 50, 50, seeds=10).mean
    single = expected_min_coverage(100, 50, 50).mean
    assert avg - single >= 0.08


def test_determinism() -> None:
    assert expected_min_coverage(10, 50, 50, reps=2000, seed=3) == expected_min_coverage(
        10, 50, 50, reps=2000, seed=3
    )
    assert seed_averaged_floor(10, 50, 50, seeds=3, reps=500, seed=4) == seed_averaged_floor(
        10, 50, 50, seeds=3, reps=500, seed=4
    )


def _power_curve(n_eval: int, coverage: float, lo: int, hi: int, q: float = 0.05) -> dict[int, float]:
    return {
        n: float(stats.binom.cdf(significance_cutoff(n_eval, n, 0.1, q), n_eval, coverage))
        for n in range(lo, hi + 1)
    }


def test_power_curve_over_calibration_sizes() -> None:
    """Power at true coverage 0.80 with 50 evaluation points, over calibration sizes 50 to 1000."""
    curve = _power_curve(50, 0.80, 50, 1000)
    sizes = sorted(curve)
    above = [n for n in sizes if curve[n] > 0.5]
    first_above = above[0]
    last_not_above = max(n for n in sizes if curve[n] <= 0.5)
    stays_above_from = last_not_above + 1
    max_power = max(curve.values())
    assert first_above < stays_above_from <= 1000
    assert any(curve[n] <= 0.5 for n in range(first_above, stays_above_from))
    assert all(curve[n] > 0.5 for n in range(stays_above_from, 1001))
    binom_cut = max(c for c in range(51) if stats.binom.cdf(c, 50, 0.9) <= 0.05)
    assert abs(max_power - float(stats.binom.cdf(binom_cut, 50, 0.80))) < 1e-12
    pc = vft.power_curve(50, 0.80, 50, 1000)
    assert pc["first_above_half"] == first_above
    assert pc["above_half_from"] == stays_above_from
    assert abs(pc["max_power"] - max_power) < 1e-12
    dips = [n for n in range(first_above, stays_above_from) if curve[n] <= 0.5]
    assert pc["not_above_half_after_first"] == dips
    assert (first_above, stays_above_from, len(dips), round(max_power, 3)) == (56, 110, 21, 0.556)
    committed = json.loads((_ROOT / "outputs" / "analysis" / "validity_floor.json").read_text())
    assert committed["power_curve"] == pc


def test_cutoff_minus_one_flags_nothing() -> None:
    """A size where no count is significant at level q returns -1, and a class with that
    evaluation size is never flagged, even with zero covered points."""
    q = 0.05
    n_cal = 50
    a, b = coverage_law(n_cal, 0.1)
    n_eval = next(n for n in range(1, 51) if stats.betabinom.cdf(0, n, a, b) > q)
    assert _brute_cutoff(n_eval, n_cal, 0.1, q) == -1
    assert significance_cutoff(n_eval, n_cal, 0.1, q) == -1
    labels = np.repeat(np.arange(2), n_eval)
    mask = np.zeros((len(labels), 2), dtype=bool)
    assert frac_significantly_undercovered(mask, labels, 2, np.array([n_cal, n_cal]), 0.1, q) == 0.0


_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = _ROOT / "scripts" / "validity_floor_table.py"
    spec = importlib.util.spec_from_file_location("validity_floor_table", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vft = _load_script()


def test_nucla_sizes_follow_the_runner_split() -> None:
    """The script derives the ntu2nucla sizes from the per-class target counts; the runner's own
    split on labels with those counts gives the same arrays, and their sums equal n_eval and
    n_pool in the committed oracle artifact."""
    from pcpbench.runner import _split_pool_and_eval

    counts = np.asarray(vft.NUCLA_TARGET_COUNTS)
    labels = np.repeat(np.arange(len(counts)), counts)
    pool_idx, eval_idx = _split_pool_and_eval(np.zeros((len(labels), len(counts))), labels, len(counts), 0)
    runner_eval = np.bincount(labels[eval_idx], minlength=len(counts))
    runner_pool = np.bincount(labels[pool_idx], minlength=len(counts))
    n_cal, n_eval = vft.nucla_sizes()
    assert n_eval.tolist() == runner_eval.tolist()
    assert n_cal.tolist() == runner_pool.tolist()
    art = json.loads((_ROOT / "outputs" / "ntu2nucla_oracle.json").read_text())
    assert int(n_eval.sum()) == int(art["n_eval"])
    assert int(n_cal.sum()) == int(art["n_pool"])
    for c, e in zip(n_cal.tolist(), n_eval.tolist()):
        assert significance_cutoff(e, c, 0.1, 0.05) == _brute_cutoff(e, c, 0.1, 0.05)


def test_imagenet_values_read_from_leaderboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ImageNet oracle means are read from the leaderboard's imagenet1k oracle rows, parsed
    independently here, and a tampered typed value fails loudly."""
    worst, frac = [], []
    for line in (_ROOT / "leaderboard.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0].startswith("imagenet1k_") and cells[1] == "oracle":
            worst.append(float(cells[3]))
            frac.append(float(cells[4]))
    got = vft.imagenet_oracle_values()
    assert got["n_rows"] == len(worst) > 0
    assert abs(got["worst_class"] - float(np.mean(worst))) < 1e-12
    assert abs(got["frac_undercovered"] - float(np.mean(frac))) < 1e-12
    vft.check_imagenet()
    monkeypatch.setattr(vft, "IMAGENET_ORACLE", 0.600)
    with pytest.raises(SystemExit):
        vft.check_imagenet()


def test_row_carries_seed_mean_z() -> None:
    row = vft.make_row("k_sweep", 10, 50, 50, 5, 2000, 0.8, "test", paper_n_averaged=25)
    fl = expected_min_coverage(10, 50, 50, reps=2000, seed=0)
    assert abs(row["z"] - (0.8 - fl.mean) / fl.sd) < 1e-12
    assert abs(row["z_seed_mean"] - (0.8 - fl.mean) / (fl.sd / np.sqrt(25))) < 1e-12
    assert row["paper_n_averaged"] == 25


def test_averaged_value_counts_follow_the_artifacts() -> None:
    """z_seed_mean divides by the number of worst-class values each paper value averages."""
    ks: dict[int, int] = {}
    for cache in vft.K_SWEEP_CACHES:
        data = json.loads((_ROOT / "outputs" / "analysis" / f"controlled_k_sweep_{cache}.json").read_text())
        for r in data["records"]:
            if r["method"] == "oracle":
                ks[int(r["k"])] = ks.get(int(r["k"]), 0) + 1
    label = json.loads((_ROOT / "outputs" / "analysis" / vft.LABEL_CACHE).read_text())
    lb = {
        name: int(json.loads((_ROOT / "outputs" / f"{name}_oracle.json").read_text())["n_valid_seeds"])
        for name in ("ntu60_xsub", "ntu2nucla")
    }
    assert set(lb.values()) == {vft.LEADERBOARD_ROW_SEEDS}
    n_im = vft.imagenet_oracle_values()["n_rows"]
    counts = vft.paper_value_counts()
    assert counts["k_sweep"] == ks
    assert counts["label_sweep"] == int(label["seeds"])
    assert counts["leaderboard"] == lb
    assert counts["imagenet1k"] == n_im * vft.LEADERBOARD_ROW_SEEDS
    assert (ks[10], ks[100], counts["label_sweep"], counts["imagenet1k"]) == (25, 25, 10, 100)
    committed = json.loads((_ROOT / "outputs" / "analysis" / "validity_floor.json").read_text())
    for row in committed["rows"]:
        if row["setting"] == "k_sweep":
            want = ks[row["K"]]
        elif row["setting"] == "label_sweep":
            want = counts["label_sweep"]
        elif row["setting"] == "imagenet1k":
            want = counts["imagenet1k"]
        elif row["setting"] in lb:
            want = lb[row["setting"]]
        else:
            assert row["paper_oracle"] is None
            continue
        assert row["paper_n_averaged"] == want
        z_mean = (row["paper_oracle"] - row["floor_mean"]) / (row["floor_sd"] / np.sqrt(want))
        assert abs(row["z_seed_mean"] - z_mean) < 1e-12


def _smallest_nondegenerate(alpha: float) -> int:
    return next(n for n in range(1, 100_000) if coverage_law(n, alpha)[1] >= 1)


def test_significance_class_status_codes() -> None:
    from pcpbench.metrics.validity_floor import significance_class_status

    s = _smallest_nondegenerate(0.1)
    n_k = np.array([0, 5, 5, 5, 5])
    n_cal = np.array([50, 0, s - 1, s, 500])
    status = significance_class_status(n_k, n_cal, 0.1)
    assert status.tolist() == [1, 2, 3, 0, 0]
    assert significance_class_status(np.array([0, 0]), np.array([0, 3]), 0.1).tolist() == [1, 1]
    s05 = _smallest_nondegenerate(0.05)
    assert s05 > s
    status05 = significance_class_status(np.array([5, 5]), np.array([s05 - 1, s05]), 0.05)
    assert status05.tolist() == [3, 0]
    with pytest.raises(ValueError):
        significance_class_status(np.array([5]), np.array([-1]), 0.1)
    with pytest.raises(ValueError):
        significance_class_status(np.array([5, 5]), np.array([50]), 0.1)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_flagged_implies_coverage_below_target(alpha: float) -> None:
    """For every non-degenerate null the q lower cutoff lies below the target count, so a flagged
    testable class always has empirical coverage below 1 - alpha."""
    q = 0.05
    s = _smallest_nondegenerate(alpha)
    cals = list(range(s, 201)) + [300, 500, 1000, 2000, 5000]
    evals = list(range(1, 61)) + [100, 137, 250, 500]
    checked = 0
    for n_cal in cals:
        for n_eval in evals:
            cutoff = significance_cutoff(n_eval, n_cal, alpha, q)
            if cutoff < 0:
                continue
            checked += 1
            assert cutoff < (1 - alpha) * n_eval, (n_cal, n_eval, cutoff)
    assert checked > 0


def test_degenerate_class_is_not_tested() -> None:
    """A class with n_cal below the smallest non-degenerate count and 49 of 50 points covered is
    excluded instead of flagged."""
    s = _smallest_nondegenerate(0.1)
    labels = np.array([0] * 50 + [1] * 50)
    mask = np.ones((100, 2), dtype=bool)
    mask[0, 0] = False
    assert frac_significantly_undercovered(mask, labels, 2, np.array([s - 1, 500])) == 0.0
    assert np.isnan(frac_significantly_undercovered(mask, labels, 2, np.array([s - 1, s - 1])))


def test_untestable_and_short_classes_pinned() -> None:
    """One untestable class (n_cal 0) and one testable class that is significantly short read
    1.0, with one untestable class counted."""
    from pcpbench.metrics.validity_floor import significance_class_status

    labels = np.array([0] * 50 + [1] * 50)
    mask = np.ones((100, 2), dtype=bool)
    mask[:30, 0] = False
    n_cal = np.array([500, 0])
    assert significance_cutoff(50, 500, 0.1, 0.05) >= 20
    assert frac_significantly_undercovered(mask, labels, 2, n_cal) == 1.0
    status = significance_class_status(np.bincount(labels, minlength=2), n_cal, 0.1)
    assert int(np.count_nonzero(status == 2)) == 1
    assert int(np.count_nonzero(status == 3)) == 0


def test_committed_sweeps_never_calibrate_below_the_smallest_nondegenerate_count() -> None:
    """The released label sweep budgets every positive m at or above the smallest non-degenerate
    count wherever worst_class is not null, and the released K sweeps calibrate on
    n_cal_per_class 50 with no shortfall, and acp and oracle on a target pool of 50 per class."""
    s = _smallest_nondegenerate(0.1)
    root = Path(__file__).resolve().parents[1] / "outputs" / "analysis"
    label = json.loads((root / "label_complexity_contrast_s3.json").read_text())
    assert label["alpha"] == 0.1
    positive = [r for r in label["per_seed"] if r["m"] > 0 and r["worst_class"] is not None]
    assert positive
    for r in positive:
        assert r["m"] >= s
        assert min(r["per_class_labeled"]) >= s, (r["method"], r["m"], r["seed"])
    sweeps = sorted(root.glob("controlled_k_sweep_*.json"))
    assert sweeps
    for path in sweeps:
        payload = json.loads(path.read_text())
        config = payload["config"]
        assert config["alpha"] == 0.1
        assert config["n_cal_per_class"] == 50, path.name
        assert config["n_cal_per_class"] >= s, path.name
        for rec in payload["records"]:
            assert rec["n_shortfall_classes"] == 0, path.name
            if rec["n_target_cal_pool"]:
                assert rec["n_target_cal_pool"] == config["n_cal_per_class"] * rec["k"]


def _enumerated_min(cal: list[int], ev: list[int], alpha: float) -> tuple[float, float]:
    laws = [coverage_law(n, alpha) for n in cal]
    pmfs = [stats.betabinom.pmf(np.arange(m + 1), m, a, b) for m, (a, b) in zip(ev, laws)]
    first = second = 0.0
    for counts in itertools.product(*(range(m + 1) for m in ev)):
        p = math.prod(float(pmf[c]) for pmf, c in zip(pmfs, counts))
        low = min(c / m for c, m in zip(counts, ev))
        first += p * low
        second += p * low * low
    return first, math.sqrt(second - first**2)


@pytest.mark.parametrize(
    "cal, ev, alpha",
    [([9, 9], [4, 4], 0.1), ([12, 30, 17], [3, 5, 4], 0.2), ([50, 20], [6, 7], 0.1)],
)
def test_exact_floor_equals_enumeration(cal: list[int], ev: list[int], alpha: float) -> None:
    exact = exact_min_coverage(len(cal), np.array(cal), np.array(ev), alpha)
    mean, sd = _enumerated_min(cal, ev, alpha)
    assert exact.mean == pytest.approx(mean, abs=1e-12)
    assert exact.sd == pytest.approx(sd, abs=1e-9)


def test_exact_floor_of_one_class_is_the_beta_binomial_mean() -> None:
    a, b = coverage_law(50, 0.1)
    assert exact_min_coverage(1, 50, 50).mean == pytest.approx(a / (a + b), abs=1e-12)


def test_exact_floor_has_the_equal_size_form_the_paper_prints() -> None:
    """(1 / m) * sum over j = 1..m of (1 - F(j - 1)) ** K with F the BetaBinomial(m, r, n + 1 -
    r) distribution function and r = ceil((n + 1) * 0.9)."""
    K, n, m = 100, 50, 50
    r = math.ceil((n + 1) * 0.9)
    assert coverage_law(n, 0.1) == (r, n + 1 - r)
    j = np.arange(1, m + 1)
    printed = float(np.sum((1.0 - stats.betabinom.cdf(j - 1, m, r, n + 1 - r)) ** K)) / m
    assert exact_min_coverage(K, n, m).mean == pytest.approx(printed, abs=1e-12)


def test_exact_floor_is_one_when_the_threshold_is_infinite() -> None:
    assert coverage_law(8, 0.1)[1] == 0
    assert exact_min_coverage(7, 8, 20) == (1.0, 0.0)


@pytest.mark.parametrize(
    "K, n_cal, n_eval",
    [(10, 50, 50), (100, 50, 50), (1000, 25, 25), (10, 500, 500), (100, 10, 50)],
)
def test_exact_floor_agrees_with_monte_carlo(K: int, n_cal: int, n_eval: int) -> None:
    mc = expected_min_coverage(K, n_cal, n_eval)
    exact = exact_min_coverage(K, n_cal, n_eval)
    assert abs(mc.mean - exact.mean) <= 4 * mc.sd / math.sqrt(mc.reps)
    assert exact.sd == pytest.approx(mc.sd, rel=0.03)


def test_exact_floor_takes_per_class_sizes() -> None:
    cal = np.array([71, 150, 73, 74, 75])
    ev = np.array([70, 150, 72, 74, 74])
    mc = expected_min_coverage(5, cal, ev)
    exact = exact_min_coverage(5, cal, ev)
    assert abs(mc.mean - exact.mean) <= 4 * mc.sd / math.sqrt(mc.reps)
    assert exact.sd == pytest.approx(mc.sd, rel=0.03)


def _levelwise_floor(cal: list[int], ev: list[int], alpha: float) -> tuple[float, float]:
    points = sorted({Fraction(j, m) for m in ev for j in range(1, m + 1)})
    survival = np.ones(len(points))
    for n, m in zip(cal, ev):
        a, b = coverage_law(n, alpha)
        if b == 0:
            continue
        ranks = np.array([math.ceil(t * m) for t in points])
        survival *= stats.betabinom.sf(ranks - 1, m, a, b)
    edges = np.array([0.0, *(float(t) for t in points)])
    mean = float(np.sum(np.diff(edges) * survival))
    second = float(np.sum(np.diff(edges**2) * survival))
    return mean, math.sqrt(second - mean**2)


def test_exact_floor_equals_the_levelwise_form_when_every_class_has_its_own_sizes() -> None:
    cal = [8, 9, 10, 19, 23, 31, 40, 50, 64, 77, 90, 137]
    ev = [5, 12, 7, 30, 11, 26, 9, 50, 33, 21, 48, 17]
    assert coverage_law(cal[0], 0.1)[1] == 0
    exact = exact_min_coverage(len(cal), np.array(cal), np.array(ev))
    mean, sd = _levelwise_floor(cal, ev, 0.1)
    assert exact.mean == pytest.approx(mean, abs=1e-12)
    assert exact.sd == pytest.approx(sd, abs=1e-9)


def test_exact_floor_stays_fast_when_every_class_has_its_own_sizes() -> None:
    """300 classes of pairwise different sizes take about a second with one survival table per
    class, where one scipy call per level and class took minutes; the bound leaves a wide margin
    on a loaded machine."""
    K = 300
    cal = np.arange(20, 20 + K)
    ev = np.arange(30, 30 + K)
    start = time.perf_counter()
    exact = exact_min_coverage(K, cal, ev)
    elapsed = time.perf_counter() - start
    mc = expected_min_coverage(K, cal, ev)
    assert abs(mc.mean - exact.mean) <= 4 * mc.sd / math.sqrt(mc.reps)
    assert exact.sd == pytest.approx(mc.sd, rel=0.03)
    assert elapsed < 30.0


def test_exact_floor_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        exact_min_coverage(0, 50, 50)
    with pytest.raises(ValueError):
        exact_min_coverage(2, 50, np.array([10, 0]))
    with pytest.raises(ValueError):
        exact_min_coverage(3, np.array([50, 50]), 10)


def test_floor_sizes_must_be_whole_numbers() -> None:
    """A size such as 50.7 is refused instead of being cut to 50 without a word."""
    with pytest.raises(ValueError, match="whole number"):
        exact_min_coverage(2, 50.7, 20)
    with pytest.raises(ValueError, match="whole number"):
        exact_min_coverage(2, 50, np.array([20.0, 20.5]))
    with pytest.raises(ValueError, match="whole number"):
        expected_min_coverage(2, 50, 20.5, reps=10)
    with pytest.raises(ValueError, match="whole number"):
        seed_averaged_floor(2, float("nan"), 20, seeds=2, reps=10)
    assert exact_min_coverage(3, 50.0, np.array([20.0, 20.0, 20.0])) == exact_min_coverage(3, 50, 20)


def test_committed_floor_rows_agree_with_the_closed_form() -> None:
    """Each row of validity_floor.json is a Monte Carlo estimate of a value the closed form gives
    without sampling."""
    payload = json.loads((_ROOT / "outputs" / "analysis" / "validity_floor.json").read_text())
    assert payload["rows"]
    for row in payload["rows"]:
        exact = exact_min_coverage(row["K"], np.asarray(row["n_cal"]), np.asarray(row["n_eval"]),
                                   payload["alpha"])
        mc_error = row["floor_sd"] / math.sqrt(row["floor_reps"])
        assert abs(row["floor_mean"] - exact.mean) <= 4 * mc_error, row["setting"]
