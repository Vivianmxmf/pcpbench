"""Regression tests for the exact-boundary rank of the conformal quantile."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pcpbench.calibrators.quantile import split_quantile, weighted_quantile


def _exact_rank(n: int, alpha: Fraction) -> int:
    level = (n + 1) * (1 - alpha)
    return -((-level.numerator) // level.denominator)


def _expected_split(n: int, rank: int) -> float:
    return float(rank) if rank <= n else float("inf")


def test_split_quantile_rank_alpha_010_every_n_to_1000() -> None:
    wrong = []
    for n in range(1, 1001):
        rank = (9 * (n + 1) + 9) // 10
        got = split_quantile(np.arange(1, n + 1), 0.1)
        if got != _expected_split(n, rank):
            wrong.append((n, got, rank))
    assert wrong == [], f"{len(wrong)} sizes off the finite-sample rank, first {wrong[:10]}"


@pytest.mark.parametrize("alpha_num, alpha_den", [(1, 20), (1, 5)])
def test_split_quantile_rank_other_alphas_to_300(alpha_num: int, alpha_den: int) -> None:
    alpha = Fraction(alpha_num, alpha_den)
    wrong = []
    for n in range(1, 301):
        rank = _exact_rank(n, alpha)
        got = split_quantile(np.arange(1, n + 1), float(alpha))
        if got != _expected_split(n, rank):
            wrong.append((n, got, rank))
    assert wrong == [], f"{len(wrong)} sizes off the finite-sample rank, first {wrong[:10]}"


def test_weighted_quantile_nonuniform_away_from_boundary_unchanged() -> None:
    rng = np.random.default_rng(12345)
    n = 200
    scores = rng.normal(size=n)
    weights = rng.uniform(0.2, 3.0, size=n)
    alpha = Fraction(1, 10)

    order = np.argsort(scores)
    w_exact = [Fraction(float(w)) for w in weights[order]]
    total = sum(w_exact) + 1
    target = 1 - alpha
    cum = Fraction(0)
    idx = None
    for i, w in enumerate(w_exact):
        cum += w / total
        if cum >= target:
            idx = i
            break
    assert idx is not None
    prev = cum - w_exact[idx] / total
    assert float(cum - target) > 1e-6 and float(target - prev) > 1e-6
    expected = float(scores[order][idx])

    assert weighted_quantile(scores, weights, float(alpha)) == expected


def test_weighted_quantile_engineered_exact_boundary_returns_that_rank() -> None:
    n = 19
    weights = np.array([2.0 if i % 2 == 0 else 1.0 for i in range(n)])
    scores = np.arange(1, n + 1, dtype=float)
    total = Fraction(int(weights.sum()) + 1)
    prefix = Fraction(int(weights[:18].sum()))
    assert prefix / total == 1 - Fraction(1, 10)

    assert weighted_quantile(scores, weights, 0.1) == 18.0
