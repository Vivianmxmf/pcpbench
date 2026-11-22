"""Label-complexity curve: coverage vs. labeled-target-points-per-class."""

from __future__ import annotations
import numpy as np
from ..calibrators.oracle import OracleTargetMondrian
from .coverage import per_class_coverage, worst_class_coverage


def label_complexity_curve(ds, alpha: float, m_grid: list[int], seed: int = 0) -> dict:
    """For each ``m`` in ``m_grid``, sample ``m`` labeled target points per class (all of them if
    a class has fewer than ``m``), fit ``OracleTargetMondrian`` on that labeled-target subset,
    and evaluate on the full target set."""
    rng = np.random.default_rng(seed)
    src_logits, src_labels = ds.source_cal()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes

    curve: dict[int, dict[str, float]] = {}
    for m in m_grid:
        sub_logits_parts = []
        sub_labels_parts = []
        for c in range(K):
            idx_c = np.where(tgt_labels == c)[0]
            if len(idx_c) == 0:
                continue
            take = min(m, len(idx_c))
            chosen = rng.choice(idx_c, size=take, replace=False)
            sub_logits_parts.append(tgt_logits[chosen])
            sub_labels_parts.append(tgt_labels[chosen])
        sub_logits = np.concatenate(sub_logits_parts)
        sub_labels = np.concatenate(sub_labels_parts)

        fitted = OracleTargetMondrian().fit(
            src_logits, src_labels,
            target_labeled=(sub_logits, sub_labels),
            alpha=alpha, seed=seed,
        )
        mask = fitted.predict_set(tgt_logits)
        curve[m] = {
            "worst_class": worst_class_coverage(mask, tgt_labels, K),
            "mean_per_class": float(np.nanmean(per_class_coverage(mask, tgt_labels, K))),
        }
    return curve
