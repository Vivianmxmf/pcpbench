"""Runner: fits a calibrator per seed, predicts on the target, and aggregates the
per-class-coverage metric row (mean + std over seeds) for the leaderboard."""

from __future__ import annotations

import numbers
from typing import Callable

import numpy as np

from .calibrators.acp import DEFAULT_TIE_BREAK as ACP_TIE_BREAK
from .calibrators.acp import ACPCalibrator
from .calibrators.base import Calibrator
from .calibrators.clustered import ClusteredMondrian
from .calibrators.mondrian import SourceMondrian
from .calibrators.oracle import OracleTargetMondrian
from .calibrators.ppi import PPIMondrian
from .calibrators.split import SplitCP
from .calibrators.weighted import WeightedCP
from .metrics import (
    frac_classes_undercovered,
    frac_significantly_undercovered,
    significance_class_status,
    marginal_coverage,
    mean_set_size,
    per_class_coverage,
    worst_class_coverage,
)
from .metrics.validity_floor import SIG_DEGENERATE_NULL, SIG_NO_CAL_POINTS
from .scores.matrix import check_score

METHODS: dict[str, Callable[[], Calibrator]] = {
    "split": SplitCP,
    "mondrian": SourceMondrian,
    "weighted": WeightedCP,
    "ppi": PPIMondrian,
    "acp": ACPCalibrator,
    "oracle": OracleTargetMondrian,
    "clustered": ClusteredMondrian,
}


def _sample_labeled_per_class(
    logits: np.ndarray, labels: np.ndarray, m: int, K: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    parts_x, parts_y = [], []
    for c in range(K):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            continue
        take = min(m, len(idx_c))
        chosen = rng.choice(idx_c, size=take, replace=False)
        parts_x.append(logits[chosen])
        parts_y.append(labels[chosen])
    if not parts_x:
        return logits[:0], labels[:0]
    return np.concatenate(parts_x), np.concatenate(parts_y)


CALIBRATION_COUNT_SOURCE: dict[str, str] = {
    "split": "source calibration set (one shared threshold, proxy)",
    "mondrian": "source calibration set",
    "weighted": "source calibration set (one shared weighted threshold, proxy)",
    "acp": "source calibration set (one shared APS threshold, proxy)",
    "clustered": "calibration part of the source calibration set (shared or "
                 "cluster threshold, proxy)",
    "oracle": "labeled target budget",
    "ppi": "labeled target budget, or the source calibration set when no budget "
           "is given (rectified threshold over the unlabeled target pool, proxy)",
}

SIG_Q = 0.05


def calibration_counts_per_class(
    method: str, fitted, src_labels: np.ndarray,
    budget_labels: np.ndarray | None, K: int,
) -> np.ndarray:
    """Per-class count of true-labeled points in the set ``method`` calibrates its threshold for
    each class on (see ``CALIBRATION_COUNT_SOURCE``)."""
    def counts(labels: np.ndarray) -> np.ndarray:
        return np.bincount(np.asarray(labels, dtype=np.int64), minlength=K)[:K]

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    has_budget = budget_labels is not None and len(budget_labels) > 0
    if method == "oracle" or (method == "ppi" and has_budget):
        if not has_budget:
            raise ValueError("oracle calibrates on a labeled target budget; none was given")
        return counts(budget_labels)
    if method == "clustered":
        return counts(np.asarray(src_labels)[fitted.calibration_mask])
    return counts(src_labels)


def _nan_to_none(values) -> list:
    return [None if isinstance(v, float) and np.isnan(v) else v for v in values]


_EVAL_FRAC = 0.5

_TARGET_LOGITS_METHODS = frozenset({"weighted", "ppi"})
_LABEL_BUDGET_METHODS = frozenset({"oracle", "ppi", "acp"})
SCORE_METHODS = frozenset({"split", "mondrian", "weighted", "ppi", "oracle", "clustered"})
ACP_SCORE = "audited_aps"
SET_RULE_SCORE_THRESHOLD = "score_threshold"
SET_RULE_GREEDY_CROSSING = "greedy_crossing"


def _split_pool_and_eval(
    tgt_logits: np.ndarray, tgt_labels: np.ndarray, K: int, seed: int,
    eval_frac: float = _EVAL_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng((seed, 2_000_000_000))
    pool_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []
    for c in range(K):
        idx_c = rng.permutation(np.where(tgt_labels == c)[0])
        if len(idx_c) == 0:
            continue
        n_eval = max(1, int(round(len(idx_c) * eval_frac)))
        n_eval = min(n_eval, len(idx_c))
        eval_parts.append(idx_c[:n_eval])
        pool_parts.append(idx_c[n_eval:])
    eval_idx = np.concatenate(eval_parts) if eval_parts else np.zeros(0, dtype=int)
    pool_idx = np.concatenate(pool_parts) if pool_parts else np.zeros(0, dtype=int)
    return pool_idx, eval_idx


def _nan_reduce(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def run(ds, method: str, alpha: float, seeds: int, m_labeled: int = 0,
        eval_frac: float = _EVAL_FRAC, score: str = "lac") -> dict:
    """Fit ``method`` for each seed in ``range(seeds)`` and score it on a genuinely held-out
    slice of the target-test set, then aggregate mean + std over seeds."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    if isinstance(seeds, bool) or not isinstance(seeds, numbers.Integral) or seeds < 1:
        raise ValueError(f"seeds must be a positive integer, got {seeds!r}")
    if isinstance(m_labeled, bool) or not isinstance(m_labeled, numbers.Integral) or m_labeled < 0:
        raise ValueError(f"m_labeled must be a non-negative integer, got {m_labeled!r}")
    if not isinstance(eval_frac, numbers.Real) or isinstance(eval_frac, bool) or not 0.0 < eval_frac <= 1.0:
        raise ValueError(f"eval_frac must be a real number in (0, 1], got {eval_frac!r}")
    if not isinstance(alpha, numbers.Real) or isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be a real number in (0, 1), got {alpha!r}")
    check_score(score)

    src_logits, src_labels = ds.source_cal()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    level = 1.0 - alpha

    per_seed: dict[str, list[float]] = {
        "marginal": [], "worst_class": [], "frac_undercovered": [], "mean_set_size": [],
    }
    n_eval_per_seed: list[int] = []
    n_pool_per_seed: list[int] = []
    n_failed = 0
    sig_per_seed: list[float | None] = []
    per_class_cov_per_seed: list[list | None] = []
    n_eval_per_class_per_seed: list[list[int]] = []
    n_cal_per_class_per_seed: list[list[int] | None] = []
    n_untestable_per_seed: list[int | None] = []
    n_degenerate_per_seed: list[int | None] = []
    valid_seed: list[bool] = []

    for seed in range(seeds):
        pool_idx, eval_idx = _split_pool_and_eval(tgt_logits, tgt_labels, K, seed, eval_frac)
        pool_logits, pool_labels = tgt_logits[pool_idx], tgt_labels[pool_idx]
        eval_logits, eval_labels = tgt_logits[eval_idx], tgt_labels[eval_idx]
        n_pool_per_seed.append(len(pool_idx))
        n_eval_per_seed.append(len(eval_idx))
        n_eval_per_class_per_seed.append(
            np.bincount(eval_labels.astype(np.int64), minlength=K)[:K].tolist()
        )

        kwargs: dict = {"alpha": alpha, "seed": seed}

        if method in _TARGET_LOGITS_METHODS:
            kwargs["target_logits"] = tgt_logits
        if method in SCORE_METHODS:
            kwargs["score"] = score

        if method in _LABEL_BUDGET_METHODS:
            if m_labeled > 0:
                rng = np.random.default_rng(seed)
                budget_logits, budget_labels = _sample_labeled_per_class(
                    pool_logits, pool_labels, m_labeled, K, rng
                )
            else:
                budget_logits, budget_labels = pool_logits, pool_labels
            if len(budget_labels) > 0:
                kwargs["target_labeled"] = (budget_logits, budget_labels)
            elif method in ("oracle", "acp"):
                n_failed += 1
                valid_seed.append(False)
                sig_per_seed.append(None)
                per_class_cov_per_seed.append(None)
                n_cal_per_class_per_seed.append(None)
                n_untestable_per_seed.append(None)
                n_degenerate_per_seed.append(None)
                continue

        fitted = METHODS[method]().fit(src_logits, src_labels, **kwargs)
        mask = fitted.predict_set(eval_logits)

        per_seed["marginal"].append(marginal_coverage(mask, eval_labels))
        per_seed["worst_class"].append(worst_class_coverage(mask, eval_labels, K))
        per_seed["frac_undercovered"].append(
            frac_classes_undercovered(mask, eval_labels, K, level)
        )
        per_seed["mean_set_size"].append(mean_set_size(mask))

        budget = kwargs.get("target_labeled")
        n_cal_k = calibration_counts_per_class(
            method, fitted, src_labels, None if budget is None else budget[1], K
        )
        valid_seed.append(True)
        sig_per_seed.append(frac_significantly_undercovered(
            mask, eval_labels, K, n_cal_k, alpha=alpha, q=SIG_Q
        ))
        per_class_cov_per_seed.append(
            _nan_to_none(per_class_coverage(mask, eval_labels, K).tolist())
        )
        n_cal_per_class_per_seed.append(n_cal_k.tolist())
        status = significance_class_status(
            np.bincount(eval_labels.astype(np.int64), minlength=K)[:K], n_cal_k, alpha
        )
        n_untestable_per_seed.append(int(np.count_nonzero(status == SIG_NO_CAL_POINTS)))
        n_degenerate_per_seed.append(int(np.count_nonzero(status == SIG_DEGENERATE_NULL)))

    row: dict = {
        "dataset": ds.meta.get("name"),
        "method": method,
        "alpha": float(alpha),
        "seeds": int(seeds),
        "n_valid_seeds": int(seeds - n_failed),
        "n_failed_seeds": int(n_failed),
        "m_labeled": int(m_labeled),
        "n_classes": int(K),
    }
    for key, values in per_seed.items():
        mean, std = _nan_reduce(values)
        row[key] = mean
        row[f"{key}_std"] = std
    row["n_eval"] = float(np.mean(n_eval_per_seed)) if n_eval_per_seed else float("nan")
    row["n_pool"] = float(np.mean(n_pool_per_seed)) if n_pool_per_seed else float("nan")

    sig_values = [float("nan") if v is None else v for v in sig_per_seed]
    row["frac_sig_undercovered"], row["frac_sig_undercovered_std"] = _nan_reduce(sig_values)
    by_seed = {key: iter(values) for key, values in per_seed.items()}
    seed_ordered: dict[str, list] = {
        "marginal": [], "worst_class": [], "frac_undercovered": [],
        "frac_sig_undercovered": [], "mean_set_size": [],
    }
    for s_idx, ok in enumerate(valid_seed):
        for key in ("marginal", "worst_class", "frac_undercovered", "mean_set_size"):
            seed_ordered[key].append(next(by_seed[key]) if ok else None)
        seed_ordered["frac_sig_undercovered"].append(sig_per_seed[s_idx])
    seed_ordered["n_sig_untestable_classes"] = n_untestable_per_seed
    seed_ordered["n_sig_degenerate_classes"] = n_degenerate_per_seed
    row["per_seed"] = {key: _nan_to_none(values) for key, values in seed_ordered.items()}
    row["per_class_coverage_per_seed"] = per_class_cov_per_seed
    row["n_eval_per_class_per_seed"] = n_eval_per_class_per_seed
    row["n_cal_per_class_per_seed"] = n_cal_per_class_per_seed
    row["n_sig_untestable_classes_per_seed"] = list(n_untestable_per_seed)
    row["n_sig_degenerate_classes_per_seed"] = list(n_degenerate_per_seed)
    if method in SCORE_METHODS:
        row["score"] = score
        row["set_rule"] = SET_RULE_SCORE_THRESHOLD
        if score != "lac":
            row["method"] = f"{method}_{score}"
    elif method == "acp":
        row["score"] = ACP_SCORE
        row["set_rule"] = SET_RULE_GREEDY_CROSSING
        row["tie_break"] = ACP_TIE_BREAK
    else:
        row["score"] = None
        row["set_rule"] = None
    return row
