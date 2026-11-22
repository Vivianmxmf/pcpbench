"""Full ``[N, K]`` nonconformity score matrices for the calibrators."""

from __future__ import annotations

import numpy as np

from .softmax_scores import lac_score_matrix, softmax

SCORES: tuple[str, ...] = ("lac", "aps")


def check_score(score: str) -> str:
    """Return ``score`` when it names a known score, else raise ``ValueError``."""
    if score not in SCORES:
        raise ValueError(f"Unknown score {score!r}; choose from {SCORES}")
    return score


def _aps_score_matrix(logits: np.ndarray) -> np.ndarray:
    p = softmax(logits)
    order = np.argsort(-p, axis=1)
    cum = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
    out = np.empty_like(p)
    np.put_along_axis(out, order, cum, axis=1)
    return out


def score_matrix(logits: np.ndarray, score: str = "lac") -> np.ndarray:
    """``[N, K]`` matrix of ``score`` for every sample and class, in float64."""
    check_score(score)
    z = np.asarray(logits, dtype=np.float64)
    if score == "lac":
        return lac_score_matrix(z)
    return _aps_score_matrix(z)
