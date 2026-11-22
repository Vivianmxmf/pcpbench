"""Clustered class-conditional calibrator (label-free) under the LAC score by default or the APS
score with ``score="aps"``."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..scores.matrix import check_score, score_matrix
from .base import sets_from_scores
from .quantile import split_quantile

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
NULL_CLUSTER = -1


def get_quantile_threshold(alpha: float) -> int:
    """Smallest n such that ``ceil((n + 1)(1 - alpha) / n) <= 1``."""
    n = 1
    while np.ceil((n + 1) * (1 - alpha) / n) > 1:
        n += 1
    return n


def default_clustering_parameters(num_classes: int, n_totalcal: float) -> tuple[int, int]:
    """(n_clustering, num_clusters) by ``get_clustering_parameters`` of Ding et al. (2023)."""
    n_clustering = int(n_totalcal * num_classes / (75 + num_classes))
    num_clusters = int(np.floor(n_clustering / 2))
    return n_clustering, num_clusters


def _finite_or_inf_quantile(scores: np.ndarray, alpha: float) -> float:
    return split_quantile(scores, alpha) if len(scores) else float("inf")


class _FittedClustered:
    def __init__(self, thr: np.ndarray, cluster_of_class: np.ndarray, n_rare: int,
                 resolved_params: dict, calibration_mask: np.ndarray,
                 clustering_skipped: bool, score: str = "lac"):
        self._thr = np.asarray(thr, dtype=float)
        self.score = score
        self.cluster_of_class = cluster_of_class
        self.n_rare = int(n_rare)
        self.resolved_params = resolved_params
        self.calibration_mask = calibration_mask
        self.clustering_skipped = bool(clustering_skipped)

    def predict_set(self, test_logits: np.ndarray) -> np.ndarray:
        return sets_from_scores(score_matrix(test_logits, self.score), self._thr)

    def thresholds(self) -> np.ndarray:
        return self._thr


def _rare_classes(y_clustering: np.ndarray, K: int, n_thresh: int) -> np.ndarray:
    counts = np.bincount(y_clustering, minlength=K)
    return counts < n_thresh


def _kmeans_assign(s: np.ndarray, y: np.ndarray, clustering: np.ndarray,
                   rare: np.ndarray, num_clusters: int,
                   levels: tuple[float, ...], seed: int) -> np.ndarray:
    K = len(rare)
    remaining = np.where(~rare)[0]
    per_class = [s[clustering & (y == c)] for c in remaining]
    emb = np.stack([np.quantile(sc, levels) for sc in per_class])
    counts = np.array([len(sc) for sc in per_class], dtype=float)
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit(
        emb, sample_weight=np.sqrt(counts))
    cluster_of_class = np.full(K, NULL_CLUSTER, dtype=int)
    cluster_of_class[remaining] = km.labels_
    return cluster_of_class


def _thresholds(s: np.ndarray, y: np.ndarray, calibration: np.ndarray,
                cluster_of_class: np.ndarray, alpha: float) -> np.ndarray:
    null_thr = _finite_or_inf_quantile(s[calibration], alpha)
    thr = np.full(len(cluster_of_class), null_thr, dtype=float)
    for k in np.unique(cluster_of_class[cluster_of_class >= 0]):
        members = np.where(cluster_of_class == k)[0]
        thr[members] = _finite_or_inf_quantile(
            s[calibration & np.isin(y, members)], alpha)
    return thr


class ClusteredMondrian:
    """Clustered conformal prediction, label-free (source calibration only)."""

    def __init__(self, n_clustering: int | None = None, num_clusters: int | None = None,
                 quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS):
        if n_clustering is not None and n_clustering < 0:
            raise ValueError(f"n_clustering must be non-negative, got {n_clustering!r}")
        if num_clusters is not None and num_clusters < 0:
            raise ValueError(f"num_clusters must be non-negative, got {num_clusters!r}")
        self.n_clustering = n_clustering
        self.num_clusters = num_clusters
        self.quantile_levels = tuple(quantile_levels)

    def fit(self, cal_logits, cal_labels, *, target_logits=None,
            target_labeled=None, alpha=0.1, seed=0,
            score: str = "lac") -> _FittedClustered:
        sm = score_matrix(cal_logits, check_score(score))
        y = np.asarray(cal_labels)
        K = sm.shape[1]
        s = sm[np.arange(len(y)), y]

        cts = np.bincount(y, minlength=K)
        n_thresh = get_quantile_threshold(alpha)
        n_min = max(int(cts.min()), n_thresh)
        num_remaining = int((cts >= n_min).sum())
        rule_ncl, rule_k = default_clustering_parameters(num_remaining, n_min)
        n_clustering = rule_ncl if self.n_clustering is None else int(self.n_clustering)
        num_clusters = rule_k if self.num_clusters is None else int(self.num_clusters)
        frac_clustering = n_clustering / n_min

        rng = np.random.default_rng(seed)
        clustering = rng.uniform(size=len(y)) < frac_clustering
        calibration = ~clustering
        rare = _rare_classes(y[clustering], K, n_thresh)
        n_rare = int(rare.sum())

        skipped = not (K - n_rare > num_clusters and num_clusters > 1)
        if skipped:
            cluster_of_class = np.full(K, NULL_CLUSTER, dtype=int)
        else:
            cluster_of_class = _kmeans_assign(s, y, clustering, rare, num_clusters,
                                              self.quantile_levels, seed)
        thr = _thresholds(s, y, calibration, cluster_of_class, alpha)

        resolved = {
            "n_thresh": n_thresh,
            "n_min": n_min,
            "num_remaining_classes": num_remaining,
            "n_clustering": n_clustering,
            "num_clusters": num_clusters,
            "frac_clustering": float(frac_clustering),
            "clustering_skipped": skipped,
            "quantile_levels": self.quantile_levels,
        }
        return _FittedClustered(thr, cluster_of_class, n_rare, resolved,
                                calibration, skipped, score)
