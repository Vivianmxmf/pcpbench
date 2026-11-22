"""Oracle target-Mondrian calibrator (non-deployable upper bound)."""

from __future__ import annotations
import numpy as np
from ..scores.matrix import check_score, score_matrix
from .quantile import split_quantile
from .base import sets_from_scores
from .mondrian import _classcond_thr


class _FittedOracle:
    def __init__(self, thr, score: str = "lac"):
        self._thr = np.asarray(thr, dtype=float)
        self.score = score

    def predict_set(self, test_logits):
        return sets_from_scores(score_matrix(test_logits, self.score), self._thr)

    def thresholds(self):
        return self._thr


class OracleTargetMondrian:
    """Per-class quantile of the selected score (LAC by default, APS with ``score="aps"``)
    calibrated directly on labeled target data."""
    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0, score="lac"):
        check_score(score)
        if target_labeled is None:
            raise ValueError(
                "OracleTargetMondrian requires target_labeled=(logits, labels)."
            )
        Lx, Ly = target_labeled
        sm = score_matrix(Lx, score)
        y = np.asarray(Ly)
        K = sm.shape[1]
        fb = split_quantile(sm[np.arange(len(y)), y], alpha)
        return _FittedOracle(_classcond_thr(sm, y, alpha, K, fb), score)
