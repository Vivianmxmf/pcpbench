"""Audited Conformal Prediction (ACP), Algorithm 1 (audited APS)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..scores.softmax_scores import (
    aps_scores,
    calibrate_quantile,
    predict_sets_from_softmax,
    softmax,
)

TIE_BREAKS = ("random", "index")
DEFAULT_TIE_BREAK = "random"
_CAL_TIE_TAG = 3_000_000_101
_TEST_TIE_TAG = 3_000_000_102


def _tie_rng(tie_break: str, seed: int, tag: int) -> np.random.Generator | None:
    if tie_break == "index":
        return None
    return np.random.default_rng((int(seed), tag))


class _AuditModel:
    def __init__(self, clf=None, scaler=None, class1_index=None,
                 constant=None):
        self._clf = clf
        self._scaler = scaler
        self._class1_index = class1_index
        self._constant = constant

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """r-hat(x) for each row of ``logits`` -> shape (N,) in [0, 1]."""
        logits = np.asarray(logits, dtype=float)
        n = logits.shape[0]
        if self._constant is not None:
            return np.full(n, self._constant, dtype=float)
        x_scaled = self._scaler.transform(logits)
        return self._clf.predict_proba(x_scaled)[:, self._class1_index]


def _fit_audit_model(logits: np.ndarray, correct: np.ndarray,
                      seed: int) -> _AuditModel:
    correct = np.asarray(correct)
    if len(np.unique(correct)) < 2:
        return _AuditModel(constant=float(correct[0]))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(logits, dtype=float))
    clf = LogisticRegression(max_iter=5000, random_state=seed)
    clf.fit(x_scaled, correct.astype(int))
    class1_index = int(np.where(clf.classes_ == 1)[0][0])
    return _AuditModel(clf=clf, scaler=scaler, class1_index=class1_index)


def _audit_probs(logits: np.ndarray, r_hat_values: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    r_hat_values = np.asarray(r_hat_values, dtype=float)
    n, k = logits.shape
    yhat = np.argmax(softmax(logits), axis=1)

    if k == 1:
        return np.ones((n, 1), dtype=float)

    other_mass = (1.0 - r_hat_values) / (k - 1)
    m = np.broadcast_to(other_mass[:, None], (n, k)).copy()
    m[np.arange(n), yhat] = r_hat_values
    return m


class _FittedACP:
    def __init__(self, audit_model: _AuditModel, k: int, threshold: float,
                 tie_break: str = "random", seed: int = 0):
        self._audit_model = audit_model
        self._k = k
        self._threshold = float(threshold)
        self._tie_break = tie_break
        self._seed = int(seed)

    def predict_set(self, test_logits: np.ndarray) -> np.ndarray:
        test_logits = np.asarray(test_logits, dtype=float)
        r_hat_test = self._audit_model.predict(test_logits)
        m_test = _audit_probs(test_logits, r_hat_test)

        pred_sets = predict_sets_from_softmax(
            m_test, self._threshold, score_type="aps", randomized=False,
            tie_rng=_tie_rng(self._tie_break, self._seed, _TEST_TIE_TAG),
        )
        n = m_test.shape[0]
        mask = np.zeros((n, self._k), dtype=bool)
        for i, included in enumerate(pred_sets):
            mask[i, included] = True
        return mask

    def thresholds(self) -> np.ndarray:
        return np.asarray(self._threshold)


class ACPCalibrator:
    """Audited Conformal Prediction, Algorithm 1 (audited APS), from Sesia et al. arXiv
    2606.14909."""

    def __init__(self, tie_break: str = DEFAULT_TIE_BREAK):
        if tie_break not in TIE_BREAKS:
            raise ValueError(
                f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")
        self._tie_break = tie_break

    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0):
        if target_labeled is None:
            raise ValueError(
                "ACPCalibrator requires target_labeled=(logits, labels) "
                "(a labeled target audit set to train the audit model "
                "r-hat)."
            )

        audit_logits, audit_labels = target_labeled
        audit_logits = np.asarray(audit_logits, dtype=float)
        audit_labels = np.asarray(audit_labels)
        yhat_audit = np.argmax(softmax(audit_logits), axis=1)
        correct = (yhat_audit == audit_labels).astype(int)
        audit_model = _fit_audit_model(audit_logits, correct, seed)

        cal_logits = np.asarray(cal_logits, dtype=float)
        cal_labels = np.asarray(cal_labels)
        k = cal_logits.shape[1]
        r_hat_cal = audit_model.predict(cal_logits)
        m_cal = _audit_probs(cal_logits, r_hat_cal)

        rng = np.random.default_rng(seed)
        scores = aps_scores(
            m_cal, cal_labels, randomized=False, rng=rng,
            tie_rng=_tie_rng(self._tie_break, seed, _CAL_TIE_TAG),
        )
        threshold = calibrate_quantile(scores, alpha)

        return _FittedACP(audit_model, k, threshold, self._tie_break, seed)
