"""Hand-built K-sweep payloads for tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

K_VALUES: tuple[int, ...] = (10, 25, 50, 100)
N_CLASSES = 100
ALL_METHODS: tuple[str, ...] = ("acp", "clustered", "mondrian", "oracle", "ppi", "split",
                                "weighted")


def cache_name(backbone: str, corruption: str, severity: int) -> str:
    middle = "vitL_" if backbone == "ViT-L/14" else ""
    return f"caches/cifar100c_{middle}{corruption}_s{severity}.npz"


def class_order(seed: int) -> list[int]:
    """A fixed class order per seed (not the sweep script's own stream)."""
    return np.random.default_rng(1000 + seed).permutation(N_CLASSES).tolist()


def make_payload(
    backbone: str, corruption: str, severity: int, *, design: str = "nested",
    n_seeds: int = 3, coverage: Callable | None = None, view: str = "shift",
    view_seed: int | None = None, cache: str | None = None, stamp: str = "abc1234",
) -> dict:
    """One sweep payload."""
    records = []
    for k in K_VALUES:
        for seed in range(n_seeds):
            ids = np.sort(np.asarray(class_order(seed)[:k]))
            for method in ALL_METHODS:
                per_class = ([0.9] * k if coverage is None
                             else [float(v) for v in coverage(method, k, seed, ids)])
                labeled = method in ("acp", "oracle")
                records.append({
                    "k": k, "subset_seed": seed, "method": method,
                    "marginal": 0.9, "worst_class": min(per_class),
                    "frac_undercovered": 0.0, "mean_set_size": 0.2 * k, "set_over_k": 0.2,
                    "n_cal_total": 50 * k, "n_shortfall_classes": 0,
                    "n_eval_total": 50 * k,
                    "n_target_cal_pool": 50 * k if labeled else 0,
                    "n_low_count_eval_classes": 0, "frac_sig_undercovered": 0.0,
                    "per_class_coverage": per_class,
                })
                if design == "nested":
                    records[-1]["n_cal_per_class"] = [30 if method == "clustered" else 50] * k
    payload = {
        "config": {"cache": cache or cache_name(backbone, corruption, severity),
                   "k_values": list(K_VALUES), "subset_seeds": n_seeds,
                   "n_cal_per_class": 50, "alpha": 0.1, "eval_frac": 0.5},
        "records": records,
        "summary": [],
    }
    if design == "nested":
        payload["config"].update({"subset_design": "nested", "view": view,
                                  "view_seed": view_seed, "method_seed": 0,
                                  "acp_tie_break": "random"})
        payload["class_orders"] = [class_order(seed) for seed in range(n_seeds)]
        payload["problems"] = [
            {"k": k, "subset_seed": seed, "class_ids": sorted(class_order(seed)[:k]),
             "n_eval_per_class": [50] * k, "eval_top1_per_class": [0.5] * k}
            for k in K_VALUES for seed in range(n_seeds)]
        payload["provenance"] = {
            "code_version": stamp,
            "python": "3.11.9", "numpy": "1.26.4", "sklearn": "1.5.1",
            "source_cal_sha256": ("b" if backbone == "ViT-L/14" else "a") * 64,
        }
    return payload


def make_acp_row(dataset: str, *, rerun: bool = False) -> dict:
    """A runner row of ACP as committed (``rerun`` False) or as the rerun writes it."""
    row = {"dataset": dataset, "method": "acp", "alpha": 0.1, "seeds": 10,
           "n_valid_seeds": 10, "n_failed_seeds": 0, "m_labeled": 0, "n_classes": 100,
           "marginal": 0.96, "worst_class": 0.56, "per_seed": {"marginal": [0.96] * 10}}
    if rerun:
        row.update({"marginal": 0.95, "worst_class": 0.81, "tie_break": "random"})
    return row


def make_default_ksweep(cache: str, *, rerun: bool = False, n_seeds: int = 2,
                        ks: tuple[int, ...] = (10, 25),
                        methods: tuple[str, ...] = ("acp", "split")) -> dict:
    """A released-schema (default design) K-sweep payload: ACP and split records, or the methods
    named."""
    records, summary = [], []
    for k in ks:
        for method in methods:
            for seed in range(n_seeds):
                acp = method == "acp"
                records.append({"k": k, "subset_seed": seed, "method": method,
                                "marginal": 0.9, "worst_class": (0.81 if rerun else 0.56) if acp else 0.7,
                                "per_class_coverage": [0.9] * k})
            summary.append({"k": k, "method": method, "n_seeds": n_seeds,
                            "worst_class_mean": (0.81 if rerun else 0.56) if method == "acp" else 0.7})
    config = {"cache": cache, "k_values": list(ks), "subset_seeds": n_seeds,
              "n_cal_per_class": 50, "alpha": 0.1, "eval_frac": 0.5}
    if rerun:
        config["acp_tie_break"] = "random"
    return {"config": config, "records": records, "summary": summary}


def make_label_sweep(*, rerun: bool = False, n_seeds: int = 2,
                     ms: tuple[int, ...] = (0, 10)) -> dict:
    per_seed, summary = [], []
    for method in ("ppi", "acp", "oracle"):
        for m in ms:
            for seed in range(n_seeds):
                acp = method == "acp"
                per_seed.append({"method": method, "m": m, "seed": seed,
                                 "worst_class": (0.8 if rerun else 0.5) if acp else 0.6})
            summary.append({"method": method, "m": m,
                            "worst_class_mean": (0.8 if rerun else 0.5) if method == "acp" else 0.6})
    payload = {"cache": "caches/cifar100c_contrast_s3.npz", "alpha": 0.1, "seeds": n_seeds,
               "m_values": list(ms), "methods": ["ppi", "acp", "oracle"], "eval_frac": 0.5,
               "per_seed": per_seed, "summary": summary,
               "eval_class_report": [{"seed": 0, "counts": [50, 50]}]}
    if rerun:
        payload["acp_tie_break"] = "random"
    return payload
