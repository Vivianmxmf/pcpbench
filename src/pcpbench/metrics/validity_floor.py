"""Validity floor of worst-class coverage under exact per-class validity."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from scipy import stats

from pcpbench.calibrators.quantile import split_quantile


@dataclass(frozen=True)
class FloorEstimate:
    K: int
    n_cal: int | None
    n_eval: int | None
    alpha: float
    reps: int
    mean: float
    sd: float
    q05: float
    q95: float


def coverage_law(n_cal: int, alpha: float) -> tuple[int, int]:
    """(a, b) of the Beta coverage law; (n_cal + 1, 0) when l == 0."""
    if n_cal < 1:
        raise ValueError("n_cal must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    q = split_quantile(np.arange(1, n_cal + 1, dtype=float), alpha)
    rank = n_cal + 1 if np.isinf(q) else int(round(q))
    l_ = n_cal + 1 - rank
    return n_cal + 1 - l_, l_


def _per_class_sizes(K: int, n: int | np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(n)
    if arr.dtype.kind not in "iu":
        as_float = arr.astype(float)
        if not np.all(np.isfinite(as_float)) or np.any(as_float != np.round(as_float)):
            raise ValueError(f"{name} must be a whole number in every class")
    if arr.ndim == 0:
        return np.full(K, int(arr), dtype=np.int64), False
    if arr.shape != (K,):
        raise ValueError(f"{name} must be a scalar or an array of length K={K}")
    return arr.astype(np.int64), True


def _draw_coverages(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    cal: np.ndarray,
    ev: np.ndarray,
    alpha: float,
) -> np.ndarray:
    laws = np.array([coverage_law(int(n), alpha) for n in cal])
    a, b = laws[:, 0].astype(float), laws[:, 1].astype(float)
    safe_b = np.where(b > 0, b, 1.0)
    p = rng.beta(a, safe_b, size=shape)
    p = np.where(b > 0, p, 1.0)
    counts = rng.binomial(ev, p)
    return counts / ev


def _summarise(
    mins: np.ndarray, K: int, n_cal: int | None, n_eval: int | None, alpha: float, reps: int
) -> FloorEstimate:
    return FloorEstimate(
        K=K,
        n_cal=n_cal,
        n_eval=n_eval,
        alpha=float(alpha),
        reps=int(reps),
        mean=float(mins.mean()),
        sd=float(mins.std(ddof=1)) if reps > 1 else 0.0,
        q05=float(np.quantile(mins, 0.05)),
        q95=float(np.quantile(mins, 0.95)),
    )


def expected_min_coverage(
    K: int,
    n_cal: int | np.ndarray,
    n_eval: int | np.ndarray,
    alpha: float = 0.1,
    reps: int = 20000,
    seed: int = 0,
) -> FloorEstimate:
    """Monte Carlo floor."""
    cal, cal_is_arr = _per_class_sizes(K, n_cal, "n_cal")
    ev, ev_is_arr = _per_class_sizes(K, n_eval, "n_eval")
    rng = np.random.default_rng(seed)
    cov = _draw_coverages(rng, (reps, K), cal, ev, alpha)
    mins = cov.min(axis=1)
    is_arr = cal_is_arr or ev_is_arr
    return _summarise(
        mins,
        K,
        None if is_arr else int(cal[0]),
        None if is_arr else int(ev[0]),
        alpha,
        reps,
    )


class ExactFloor(NamedTuple):
    """Mean and standard deviation of the floor, computed without sampling."""

    mean: float
    sd: float


def exact_min_coverage(
    K: int,
    n_cal: int | np.ndarray,
    n_eval: int | np.ndarray,
    alpha: float = 0.1,
) -> ExactFloor:
    """The floor that ``expected_min_coverage`` estimates, in closed form."""
    if K < 1:
        raise ValueError("K must be at least 1")
    cal, _ = _per_class_sizes(K, n_cal, "n_cal")
    ev, _ = _per_class_sizes(K, n_eval, "n_eval")
    if np.any(ev < 1):
        raise ValueError("n_eval must be at least 1 in every class")
    by_eval: dict[int, Counter[int]] = {}
    for n, m in zip(cal.tolist(), ev.tolist()):
        by_eval.setdefault(m, Counter())[n] += 1
    points = sorted({Fraction(j, m) for m in by_eval for j in range(1, m + 1)})
    numer = np.array([t.numerator for t in points], dtype=np.int64)
    denom = np.array([t.denominator for t in points], dtype=np.int64)
    survival = np.ones(len(points))
    for m, calibration_sizes in by_eval.items():
        ranks = -((-numer * m) // denom)
        for n, count in calibration_sizes.items():
            table = _survival_table(n, m, alpha)
            if table is not None:
                survival *= table[ranks] ** count
    edges = np.concatenate(([0.0], numer / denom))
    mean = float(np.sum(np.diff(edges) * survival))
    second = float(np.sum(np.diff(edges**2) * survival))
    return ExactFloor(mean=mean, sd=math.sqrt(max(second - mean**2, 0.0)))


def _survival_table(n_cal: int, n_eval: int, alpha: float) -> np.ndarray | None:
    a, b = coverage_law(n_cal, alpha)
    if b == 0:
        return None
    pmf = stats.betabinom.pmf(np.arange(n_eval + 1), n_eval, a, b)
    return np.minimum(np.cumsum(pmf[::-1])[::-1], 1.0)


def seed_averaged_floor(
    K: int,
    n_cal: int | np.ndarray,
    n_eval: int | np.ndarray,
    seeds: int,
    alpha: float = 0.1,
    reps: int = 5000,
    seed: int = 0,
) -> FloorEstimate:
    """Per-class empirical coverage averaged over `seeds` independent calibration and evaluation
    draws, then the minimum over K classes."""
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    cal, cal_is_arr = _per_class_sizes(K, n_cal, "n_cal")
    ev, ev_is_arr = _per_class_sizes(K, n_eval, "n_eval")
    rng = np.random.default_rng(seed)
    cov = _draw_coverages(rng, (reps, seeds, K), cal, ev, alpha)
    mins = cov.mean(axis=1).min(axis=1)
    is_arr = cal_is_arr or ev_is_arr
    return _summarise(
        mins,
        K,
        None if is_arr else int(cal[0]),
        None if is_arr else int(ev[0]),
        alpha,
        reps,
    )


def undercovered_false_positive_rate(
    n_cal: int, n_eval: int, alpha: float = 0.1, level: float = 0.9
) -> float:
    """P(empirical class coverage < level) under exact validity, computed exactly with
    scipy.stats.betabinom (b > 0) or as 0.0 when l == 0."""
    a, b = coverage_law(n_cal, alpha)
    if b == 0:
        return 0.0
    c = int(np.ceil(level * n_eval - 1e-12)) - 1
    if c < 0:
        return 0.0
    return float(stats.betabinom.cdf(c, n_eval, a, b))


def significance_cutoff(n_eval: int, n_cal: int, alpha: float = 0.1, q: float = 0.05) -> int:
    """Largest count c with P(BetaBinomial(n_eval, a, b) <= c) <= q, where (a, b) is
    coverage_law(n_cal, alpha): the one-sided null of a valid per-class calibration on n_cal
    points."""
    a, b = coverage_law(n_cal, alpha)
    if b == 0:
        return n_eval - 1
    cdf = stats.betabinom.cdf(np.arange(n_eval + 1), n_eval, a, b)
    return int(np.count_nonzero(cdf <= q)) - 1


SIG_TESTABLE = 0
SIG_NO_EVAL_POINTS = 1
SIG_NO_CAL_POINTS = 2
SIG_DEGENERATE_NULL = 3


def significance_class_status(
    n_k: np.ndarray, n_cal_per_class: np.ndarray, alpha: float = 0.1
) -> np.ndarray:
    """Per-class status of the significance test behind ``frac_significantly_undercovered``, as
    small integer codes."""
    n_k = np.asarray(n_k, dtype=np.int64)
    n_cal = np.asarray(n_cal_per_class, dtype=np.int64)
    if n_k.shape != n_cal.shape or n_k.ndim != 1:
        raise ValueError("n_k and n_cal_per_class must be one-dimensional arrays of equal length")
    if np.any(n_cal < 0) or np.any(n_k < 0):
        raise ValueError("counts must be non-negative")
    status = np.full(n_k.shape, SIG_TESTABLE, dtype=np.int64)
    laws: dict[int, int] = {}
    for c in range(len(n_k)):
        if n_k[c] == 0:
            status[c] = SIG_NO_EVAL_POINTS
        elif n_cal[c] == 0:
            status[c] = SIG_NO_CAL_POINTS
        else:
            n = int(n_cal[c])
            if n not in laws:
                laws[n] = coverage_law(n, alpha)[1]
            if laws[n] < 1:
                status[c] = SIG_DEGENERATE_NULL
    return status


def frac_significantly_undercovered(
    mask: np.ndarray,
    labels: np.ndarray,
    K: int,
    n_cal_per_class: np.ndarray,
    alpha: float = 0.1,
    q: float = 0.05,
) -> float:
    """Fraction of testable classes whose covered count is at most significance_cutoff(n_eval_k,
    n_cal_k, alpha, q); NaN when no class is testable."""
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask)
    n_cal_arr = np.asarray(n_cal_per_class, dtype=np.int64)
    if n_cal_arr.shape != (K,):
        raise ValueError(f"n_cal_per_class must have length K={K}")
    if len(labels):
        covered = mask[np.arange(len(labels)), labels].astype(float)
        n_k = np.bincount(labels, minlength=K)[:K]
        hits = np.bincount(labels, weights=covered, minlength=K)[:K]
    else:
        n_k, hits = np.zeros(K, np.int64), np.zeros(K)
    testable = np.flatnonzero(
        significance_class_status(n_k, n_cal_arr, alpha) == SIG_TESTABLE
    )
    if testable.size == 0:
        return float("nan")
    flagged = [
        hits[c] <= significance_cutoff(int(n_k[c]), int(n_cal_arr[c]), alpha, q)
        for c in testable
    ]
    return float(np.mean(flagged))
