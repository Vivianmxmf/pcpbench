"""Weighted-CP calibrator with a generic (logits-in) density-ratio weighter."""

from __future__ import annotations
import numpy as np
from ..scores.matrix import check_score, score_matrix
from .quantile import weighted_quantile

EPS = 1e-12


class _DensityRatioWeighter:
    def __init__(self, clip=20.0, C=1.0, seed=0):
        self.clip, self.C, self.seed = clip, C, seed
        self._clf = self._scaler = None
        self._tcol = self._scol = None

    def fit(self, source_x, target_x):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        x = np.vstack([source_x, target_x])
        y = np.concatenate([np.zeros(len(source_x)), np.ones(len(target_x))])
        self._scaler = StandardScaler().fit(x)
        self._clf = LogisticRegression(max_iter=5000, C=self.C,
                                       class_weight="balanced",
                                       random_state=self.seed).fit(
            self._scaler.transform(x), y)
        cls = list(self._clf.classes_)
        self._tcol, self._scol = cls.index(1.0), cls.index(0.0)
        return self

    def weights(self, x):
        p = self._clf.predict_proba(self._scaler.transform(x))
        return np.clip(p[:, self._tcol] / np.clip(p[:, self._scol], EPS, None),
                       0.0, self.clip)


class _FittedWeighted:
    def __init__(self, cal_scores, w_cal, weighter, alpha, thr_global,
                 score: str = "lac"):
        self._cal_scores, self._w_cal = cal_scores, w_cal
        self._weighter, self._alpha = weighter, alpha
        self._thr_global = thr_global
        self.score = score

    def predict_set(self, test_logits):
        sm = score_matrix(test_logits, self.score)
        w_test = self._weighter.weights(test_logits)
        thr = np.array([
            weighted_quantile(self._cal_scores, self._w_cal / max(w_test[j], EPS),
                              self._alpha)
            for j in range(len(sm))
        ])
        return sm <= thr[:, None]

    def thresholds(self):
        return np.asarray(self._thr_global)


class WeightedCP:
    """Weighted conformal: a density-ratio weighted quantile of the selected score (LAC by
    default, APS with ``score="aps"``) over the source calibration set, one threshold per test
    point."""

    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0, score="lac"):
        check_score(score)
        if target_logits is None:
            raise ValueError("WeightedCP requires target_logits (unlabeled target).")
        y = np.asarray(cal_labels)
        cal_scores = score_matrix(cal_logits, score)[np.arange(len(y)), y]
        w = _DensityRatioWeighter(seed=seed).fit(cal_logits, target_logits)
        w_cal = w.weights(cal_logits)
        thr_global = weighted_quantile(cal_scores, w_cal, alpha)
        return _FittedWeighted(cal_scores, w_cal, w, alpha, thr_global, score)
