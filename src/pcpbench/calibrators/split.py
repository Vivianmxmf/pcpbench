from __future__ import annotations
import numpy as np
from ..scores.matrix import check_score, score_matrix
from .quantile import split_quantile
from .base import sets_from_scores


class _FittedSplit:
    def __init__(self, thr: float, score: str = "lac"):
        self._thr = float(thr)
        self.score = score

    def predict_set(self, test_logits):
        return sets_from_scores(score_matrix(test_logits, self.score), self._thr)

    def thresholds(self):
        return np.asarray(self._thr)


class SplitCP:
    """Split conformal: one shared quantile of the selected score (LAC by default, APS with
    ``score="aps"``) over the source calibration set."""

    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0, score="lac"):
        check_score(score)
        y = np.asarray(cal_labels)
        s = score_matrix(cal_logits, score)[np.arange(len(y)), y]
        return _FittedSplit(split_quantile(s, alpha), score)
