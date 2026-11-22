"""Softmax-based conformal prediction baselines: LAC, APS, RAPS."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as _scipy_softmax


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to softmax probabilities (numerically stable)."""
    return _scipy_softmax(logits, axis=-1)


def descending_order(
    probs: np.ndarray,
    tie_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Column indices of each row of ``probs`` by descending probability."""
    if tie_rng is None:
        return np.argsort(-probs, axis=1)
    return np.lexsort((tie_rng.random(probs.shape), -probs), axis=1)


def lac_scores(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """LAC (Least Ambiguous Classification) nonconformity scores."""
    n = len(labels)
    return 1.0 - softmax_probs[np.arange(n), labels]


def aps_scores(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
    randomized: bool = True,
    rng: np.random.Generator | None = None,
    tie_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """APS (Adaptive Prediction Sets) nonconformity scores."""
    N, C = softmax_probs.shape
    sorted_idx = descending_order(softmax_probs, tie_rng)
    sorted_probs = np.take_along_axis(softmax_probs, sorted_idx, axis=1)
    cumsum_probs = np.cumsum(sorted_probs, axis=1)

    scores = np.empty(N)
    _rng = rng if rng is not None else np.random.default_rng()

    for i in range(N):
        y = labels[i]
        rank = int(np.where(sorted_idx[i] == y)[0][0])
        s = cumsum_probs[i, rank]
        if randomized:
            s = s - _rng.uniform(0.0, sorted_probs[i, rank])
        scores[i] = s

    return scores


def raps_scores(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
    lam: float = 0.01,
    k_reg: int = 5,
    randomized: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """RAPS (Regularized APS) nonconformity scores."""
    N, C = softmax_probs.shape
    sorted_idx = np.argsort(-softmax_probs, axis=1)
    sorted_probs = np.take_along_axis(softmax_probs, sorted_idx, axis=1)
    cumsum_probs = np.cumsum(sorted_probs, axis=1)

    scores = np.empty(N)
    _rng = rng if rng is not None else np.random.default_rng()

    for i in range(N):
        y = labels[i]
        rank = int(np.where(sorted_idx[i] == y)[0][0])
        s = cumsum_probs[i, rank]
        if randomized:
            s = s - _rng.uniform(0.0, sorted_probs[i, rank])
        penalty = lam * max(rank + 1 - k_reg, 0)
        scores[i] = s + penalty

    return scores


def calibrate_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Compute the (1-alpha) conformal quantile from calibration scores."""
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def predict_sets_from_softmax(
    softmax_probs: np.ndarray,
    threshold: float,
    score_type: str = "aps",
    lam: float = 0.01,
    k_reg: int = 5,
    randomized: bool = False,
    rng: np.random.Generator | None = None,
    tie_rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Build prediction sets from softmax probabilities and a calibrated threshold."""
    N, C = softmax_probs.shape
    sorted_idx = descending_order(softmax_probs, tie_rng)
    sorted_probs = np.take_along_axis(softmax_probs, sorted_idx, axis=1)

    prediction_sets: list[list[int]] = []

    if score_type == "lac":
        for i in range(N):
            included = [c for c in range(C) if softmax_probs[i, c] >= 1.0 - threshold]
            prediction_sets.append(included)
        return prediction_sets

    _rng = rng if rng is not None else np.random.default_rng()
    for i in range(N):
        cumsum = 0.0
        included: list[int] = []
        for rank, c in enumerate(sorted_idx[i]):
            p = sorted_probs[i, rank]
            u = _rng.uniform(0.0, p) if randomized else 0.0
            penalty = lam * max(rank + 1 - k_reg, 0) if score_type == "raps" else 0.0
            cumsum += p - u + penalty
            included.append(int(c))
            if cumsum >= threshold:
                break
        prediction_sets.append(included)

    return prediction_sets


def lac_score_matrix(logits: np.ndarray) -> np.ndarray:
    """Full LAC score matrix: [i,c] = 1 - softmax(logits)[i,c]."""
    return 1.0 - softmax(np.asarray(logits, dtype=float))
