"""Shared weighted-quantile calibration primitive."""

from __future__ import annotations

import numpy as np


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    """Compute the weighted (1 - alpha) quantile for conformal calibration."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.min() < 0:
        raise ValueError("Importance weights must be non-negative")

    w_total = weights.sum() + 1.0
    w_norm = weights / w_total

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_w = w_norm[order]

    cumulative = np.cumsum(sorted_w)
    idx = np.searchsorted(cumulative, 1.0 - alpha - 1e-9, side="left")
    if idx >= len(sorted_scores):
        return float("inf")
    return float(sorted_scores[idx])


def split_quantile(scores: np.ndarray, alpha: float) -> float:
    """Unweighted split-CP quantile."""
    return float(weighted_quantile(scores, np.ones(len(scores)), alpha=alpha))
