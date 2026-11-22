from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


def sets_from_scores(score_matrix: np.ndarray, thr) -> np.ndarray:
    """Bool [N,K] prediction mask: class c in set iff score[i,c] <= thr."""
    return np.asarray(score_matrix) <= np.asarray(thr)


@runtime_checkable
class FittedCalibrator(Protocol):
    def predict_set(self, test_logits: np.ndarray) -> np.ndarray: ...
    def thresholds(self) -> np.ndarray: ...


@runtime_checkable
class Calibrator(Protocol):
    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha: float = 0.1,
            seed: int = 0, score: str = "lac") -> FittedCalibrator: ...
