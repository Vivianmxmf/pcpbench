from __future__ import annotations
import numpy as np


def marginal_coverage(mask, labels):
    labels = np.asarray(labels)
    return float(mask[np.arange(len(labels)), labels].mean())


def per_class_coverage(mask, labels, K):
    labels = np.asarray(labels)
    out = np.full(K, np.nan)
    for c in range(K):
        m = labels == c
        if m.any():
            out[c] = float(mask[np.where(m)[0], c].mean())
    return out


def worst_class_coverage(mask, labels, K):
    return float(np.nanmin(per_class_coverage(mask, labels, K)))


def frac_classes_undercovered(mask, labels, K, level):
    pc = per_class_coverage(mask, labels, K)
    valid = ~np.isnan(pc)
    return float((pc[valid] < level).mean())


def mean_set_size(mask):
    return float(np.asarray(mask).sum(1).mean())
