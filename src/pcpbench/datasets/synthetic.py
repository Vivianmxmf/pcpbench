from __future__ import annotations
import numpy as np
from .shift_dataset import ShiftDataset, SOURCE_CAL, TARGET_TEST


def make_synthetic_shift(K: int = 12, n_per_class: int = 500,
                         shift: float = 0.6, noise: float = 0.3,
                         seed: int = 0) -> ShiftDataset:
    """Deterministic synthetic logits with per-class collapse under shift."""
    rng = np.random.default_rng(seed)
    sep = 3.0
    n_hard = max(1, K // 4)

    def gen(scale_shift: float):
        logits, labels = [], []
        for c in range(K):
            mu = np.zeros(K)
            mu[c] = sep
            if scale_shift > 0 and c < n_hard:
                mu[c] -= sep * scale_shift
                mu[(c + 1) % K] += sep * scale_shift * 0.5
            sigma = noise + scale_shift * 0.5
            x = mu + rng.normal(0.0, sigma, size=(n_per_class, K))
            logits.append(x)
            labels.append(np.full(n_per_class, c))
        return np.concatenate(logits), np.concatenate(labels)

    src_logits, src_labels = gen(0.0)
    tgt_logits, tgt_labels = gen(shift)

    logits = np.concatenate([src_logits, tgt_logits]).astype(np.float32)
    labels = np.concatenate([src_labels, tgt_labels]).astype(np.int64)
    split = np.concatenate([
        np.full(len(src_labels), SOURCE_CAL, dtype=np.int8),
        np.full(len(tgt_labels), TARGET_TEST, dtype=np.int8),
    ])
    domain = np.concatenate([
        np.zeros(len(src_labels), dtype=np.int32),
        np.ones(len(tgt_labels), dtype=np.int32),
    ])
    meta = {"name": "synthetic_shift", "shift": shift, "noise": noise,
            "seed": seed, "shift_family": "synthetic", "n_classes": K}
    return ShiftDataset(logits=logits, labels=labels, split=split,
                        domain=domain, meta=meta)
