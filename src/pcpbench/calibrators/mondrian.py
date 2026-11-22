"""Source Mondrian (class-conditional) calibrator under the LAC score by default or the APS score
with ``score="aps"``."""

from __future__ import annotations
import numpy as np
from ..scores.matrix import check_score, score_matrix
from .quantile import split_quantile
from .base import sets_from_scores


def _classcond_thr(score_matrix, labels, alpha, K, fallback):
    thr = np.full(K, fallback, dtype=float)
    for c in range(K):
        s = score_matrix[labels == c, c]
        if len(s) >= 1:
            thr[c] = split_quantile(s, alpha)
    return thr


class _FittedMondrian:
    def __init__(self, thr, score: str = "lac"):
        self._thr = np.asarray(thr, dtype=float)
        self.score = score

    def predict_set(self, test_logits):
        return sets_from_scores(score_matrix(test_logits, self.score), self._thr)

    def thresholds(self):
        return self._thr


class SourceMondrian:
    """Per-class quantile of the selected score (LAC by default, APS with ``score="aps"``) on the
    source calibration set, with the global quantile as the fallback for a class without
    calibration points."""

    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0, score="lac"):
        sm = score_matrix(cal_logits, check_score(score))
        y = np.asarray(cal_labels)
        K = sm.shape[1]
        fb = split_quantile(sm[np.arange(len(y)), y], alpha)
        return _FittedMondrian(_classcond_thr(sm, y, alpha, K, fb), score)
