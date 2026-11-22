"""PPI (prediction-powered inference) class-conditional quantile calibrator."""

from __future__ import annotations

import numpy as np

from .quantile import split_quantile as _q


def _naive_thr(sL_c: np.ndarray, alpha: float, min_needed: int, fallback: float) -> float:
    if len(sL_c) < min_needed:
        return float(fallback)
    q = _q(sL_c, alpha)
    return float(q) if np.isfinite(q) else float(fallback)


def _isotonic(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    means: list[float] = []
    counts: list[int] = []
    for v in y:
        means.append(float(v))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            c2, m2 = counts.pop(), means.pop()
            c1, m1 = counts.pop(), means.pop()
            c = c1 + c2
            means.append((m1 * c1 + m2 * c2) / c)
            counts.append(c)
    out = np.empty(n, dtype=float)
    pos = 0
    for m, c in zip(means, counts):
        out[pos:pos + c] = m
        pos += c
    return out


def _ppi_thr(s_pool_c: np.ndarray, py_pool_c: np.ndarray,
             sL_c: np.ndarray, yL_c: np.ndarray, pyL_c: np.ndarray,
             alpha: float, M: int, fallback: float) -> float:
    nP = len(s_pool_c)
    if nP == 0:
        return float(fallback)
    termB_P = float(py_pool_c.mean())
    rectB = (float(yL_c.sum()) - float(pyL_c.sum())) / M if M > 0 else 0.0
    Bhat = termB_P + rectB
    if Bhat <= 1e-9:
        return float(fallback)
    target = (1.0 - alpha) * Bhat

    parts = [s_pool_c[py_pool_c]]
    if M > 0:
        parts += [sL_c[yL_c > 0], sL_c[pyL_c > 0]]
    parts = [p for p in parts if len(p)]
    if not parts:
        return float(fallback)
    cand = np.unique(np.concatenate(parts))

    pool_ps = np.sort(s_pool_c[py_pool_c])
    lab_y = np.sort(sL_c[yL_c > 0]) if M > 0 else np.array([])
    lab_py = np.sort(sL_c[pyL_c > 0]) if M > 0 else np.array([])

    termP = np.searchsorted(pool_ps, cand, side="right") / nP
    rect = ((np.searchsorted(lab_y, cand, side="right")
             - np.searchsorted(lab_py, cand, side="right")) / M) if M > 0 else 0.0
    A = _isotonic(termP + rect)

    ok = np.flatnonzero(A - target >= -1e-12)
    return float(cand[ok[0]]) if ok.size else float(fallback)


def _rho(t_scores: np.ndarray, t_lbl: np.ndarray, K: int) -> np.ndarray:
    yhat = t_scores.argmin(axis=1)
    rho = np.full(K, np.nan)
    for c in range(K):
        m = t_lbl == c
        if m.any():
            rho[c] = float((yhat[m] != c).mean())
    return rho


class _FittedPPI:
    def __init__(self, thr, score: str = "lac"):
        self._thr = np.asarray(thr, dtype=float)
        self.score = score

    def predict_set(self, test_logits):
        from ..scores.matrix import score_matrix
        return score_matrix(test_logits, self.score) <= self._thr

    def thresholds(self):
        return self._thr


class PPIMondrian:
    """Per-class PPI quantile: unlabeled target pool + small labeled budget."""
    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0, score="lac"):
        from ..scores.matrix import check_score, score_matrix
        check_score(score)
        if target_logits is None:
            raise ValueError("PPIMondrian requires target_logits (unlabeled pool).")
        pool_s = score_matrix(target_logits, score)
        pool_pred = score_matrix(target_logits, "lac").argmin(1)
        K = pool_s.shape[1]
        fb = _q(score_matrix(cal_logits, score)[np.arange(len(cal_labels)),
                                                np.asarray(cal_labels)], alpha)
        if target_labeled is not None:
            Lx, Ly = target_labeled
            lab_s = score_matrix(Lx, score)
            lab_pred = score_matrix(Lx, "lac").argmin(1)
            M = len(Ly)
        else:
            lab_s = np.empty((0, K)); lab_pred = np.empty(0, int)
            Ly = np.empty(0, int); M = 0
        thr = np.full(K, fb, dtype=float)
        for c in range(K):
            s_pool_c = pool_s[:, c]; py_pool_c = pool_pred == c
            if M > 0:
                sL_c = lab_s[:, c]
                yL_c = (np.asarray(Ly) == c).astype(float)
                pyL_c = (lab_pred == c).astype(float)
            else:
                sL_c = np.empty(0); yL_c = np.empty(0); pyL_c = np.empty(0)
            thr[c] = _ppi_thr(s_pool_c, py_pool_c, sL_c, yL_c, pyL_c,
                              alpha, M, fb)
        return _FittedPPI(thr, score)
