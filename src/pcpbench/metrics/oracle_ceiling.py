"""Oracle 2-D PPI++ label-efficiency ceiling."""

from __future__ import annotations
import numpy as np


_NAN_ORACLE_GAIN = {"a": float("nan"), "b": float("nan"), "var_u": float("nan"),
                    "r2_scalar": float("nan"), "r2_multi": float("nan"),
                    "gain_scalar": float("nan"), "gain_multi": float("nan")}


def _oracle_gain(s_c: np.ndarray, true_c: np.ndarray, pseudo_c: np.ndarray, q_c: float,
                 alpha: float) -> dict:
    z = s_c <= q_c
    gA = (z & true_c).astype(float)
    fA = (z & pseudo_c).astype(float)
    gB = true_c.astype(float)
    fB = pseudo_c.astype(float)
    a = float(gA.mean())
    b = float(gB.mean())
    if b < 1e-9:
        return dict(_NAN_ORACLE_GAIN)
    u = gA / b - (a / b ** 2) * gB
    var_u = float(u.var())
    if var_u < 1e-12:
        return dict(_NAN_ORACLE_GAIN)

    v = fA - (1.0 - alpha) * fB
    vv = float(v.var())
    corr2 = float(np.cov(u, v, bias=True)[0, 1] ** 2 / (var_u * vv)) if vv > 1e-12 else 0.0
    corr2 = min(max(corr2, 0.0), 1.0 - 1e-9)

    x = np.column_stack([np.ones_like(fA), fA, fB])
    beta, _, _, _ = np.linalg.lstsq(x, u, rcond=None)
    resid = u - x @ beta
    r2 = float(1.0 - (resid @ resid) / (var_u * len(u)))
    r2 = min(max(r2, 0.0), 1.0 - 1e-9)

    return {"a": a, "b": b, "var_u": var_u, "r2_scalar": corr2, "r2_multi": r2,
            "gain_scalar": 1.0 / (1.0 - corr2), "gain_multi": 1.0 / (1.0 - r2)}


def oracle_ceiling(target_logits: np.ndarray, target_labels: np.ndarray,
                    alpha: float = 0.1) -> np.ndarray:
    """Per-class best-case PPI++ label-efficiency gain."""
    from ..scores.softmax_scores import lac_score_matrix
    from ..calibrators.quantile import split_quantile

    sm = lac_score_matrix(target_logits)
    y = np.asarray(target_labels)
    K = sm.shape[1]
    pred = sm.argmin(axis=1)
    gains = np.full(K, np.nan)
    for c in range(K):
        mask_c = (y == c)
        if not mask_c.any():
            continue
        q_c = split_quantile(sm[mask_c, c], alpha)
        if not np.isfinite(q_c):
            continue
        g = _oracle_gain(sm[:, c], y == c, pred == c, float(q_c), alpha)
        gains[c] = g["gain_multi"]
    return gains
