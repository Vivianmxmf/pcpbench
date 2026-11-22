#!/usr/bin/env python3
"""Controlled K-sweep: isolate the worst-of-K effect on ONE dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sklearn

from pcpbench.calibrators.acp import DEFAULT_TIE_BREAK as ACP_TIE_BREAK
from pcpbench.datasets.shift_dataset import SOURCE_CAL, TARGET_TEST, ShiftDataset
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.coverage import (
    frac_classes_undercovered,
    marginal_coverage,
    mean_set_size,
    per_class_coverage,
    worst_class_coverage,
)
from pcpbench.metrics.validity_floor import frac_significantly_undercovered
from pcpbench.runner import SIG_Q, METHODS, calibration_counts_per_class, run

LABELED_TARGET_METHODS = frozenset({"acp", "oracle"})
EVAL_COUNT_FLOOR = 5
DEFAULT_EVAL_FRAC = 0.5

SUBSET_DESIGNS = ("independent", "nested")
VIEWS = ("shift", "no_shift")
NESTED_ORDER_TAG = 3_000_000_001
NESTED_SOURCE_TAG = 3_000_000_002
NESTED_TARGET_TAG = 3_000_000_003
NO_SHIFT_VIEW_TAG = 3_000_000_004
NO_SHIFT_TARGET_PER_CLASS = 100


@dataclass(frozen=True)
class SweepRecord:
    """One (K, subset_seed, method) cell of the controlled sweep."""

    k: int
    subset_seed: int
    method: str
    marginal: float
    worst_class: float
    frac_undercovered: float
    mean_set_size: float
    set_over_k: float
    n_cal_total: int
    n_shortfall_classes: int
    n_eval_total: int
    n_target_cal_pool: int
    n_low_count_eval_classes: int
    frac_sig_undercovered: float
    per_class_coverage: tuple[float | None, ...]


class SweepSummary(NamedTuple):
    """Mean +/- std over the R subset seeds for one (K, method) pair."""

    k: int
    method: str
    n_seeds: int
    worst_class_mean: float
    worst_class_std: float
    marginal_mean: float
    marginal_std: float
    set_over_k_mean: float
    set_over_k_std: float
    frac_sig_undercovered_mean: float
    frac_sig_undercovered_std: float


def nested_class_order(n_total_classes: int, seed: int) -> np.ndarray:
    """The class order of one subset seed under the nested design: a uniformly random permutation
    of the class ids."""
    return np.random.default_rng((seed, NESTED_ORDER_TAG)).permutation(n_total_classes)


def draw_class_subset(
    k: int, n_total_classes: int, seed: int, design: str = "independent",
) -> np.ndarray:
    """Draw a uniformly random, reproducible subset of ``k`` distinct class ids from
    ``range(n_total_classes)``."""
    if k > n_total_classes:
        raise ValueError(f"k={k} exceeds available classes ({n_total_classes}).")
    if design == "nested":
        return np.sort(nested_class_order(n_total_classes, seed)[:k])
    if design != "independent":
        raise ValueError(f"unknown subset design {design!r}; expected one of {SUBSET_DESIGNS}.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total_classes, size=k, replace=False))


def restrict_to_subset(ds: ShiftDataset, subset: np.ndarray) -> ShiftDataset:
    """Restrict both splits to rows whose true label is in ``subset`` and the logit matrix to the
    K columns indexed by ``subset``."""
    mask = np.isin(ds.labels, subset)
    new_labels = np.searchsorted(subset, ds.labels[mask]).astype(np.int64)
    return ShiftDataset(
        logits=ds.logits[mask][:, subset],
        labels=new_labels,
        split=ds.split[mask],
        domain=ds.domain[mask],
        meta=dict(ds.meta),
    )


def controlled_calibration_indices(
    labels: np.ndarray, split: np.ndarray, k: int, n_per_class: int, seed: int,
    class_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Row indices into an already-restricted dataset keeping ALL target rows plus EXACTLY
    ``n_per_class`` source-calibration rows per class (all available rows if a class has fewer --
    never padded)."""
    rng = np.random.default_rng(seed)
    shortfalls: dict[int, int] = {}
    chosen: list[np.ndarray] = []
    for c in range(k):
        idx_c = np.where((split == SOURCE_CAL) & (labels == c))[0]
        if len(idx_c) < n_per_class:
            shortfalls[c] = n_per_class - len(idx_c)
            chosen.append(idx_c)
        else:
            rng_c = rng if class_ids is None else np.random.default_rng(
                (seed, NESTED_SOURCE_TAG, int(class_ids[c]) + 1))
            chosen.append(rng_c.choice(idx_c, size=n_per_class, replace=False))
    target_idx = np.where(split == TARGET_TEST)[0]
    indices = np.concatenate(chosen + [target_idx])
    return indices, shortfalls


def build_k_subproblem(
    ds: ShiftDataset, k: int, subset_seed: int, n_cal_per_class: int,
    design: str = "independent",
) -> tuple[ShiftDataset, dict[int, int]]:
    """Full restriction pipeline for one (K, subset_seed): draw the class subset, restrict
    rows/columns to it, then control per-class calibration size."""
    subset = draw_class_subset(k, ds.n_classes, subset_seed, design)
    restricted = restrict_to_subset(ds, subset)
    idx, shortfalls = controlled_calibration_indices(
        restricted.labels, restricted.split, k, n_cal_per_class, subset_seed,
        class_ids=subset if design == "nested" else None,
    )
    sub = ShiftDataset(
        logits=restricted.logits[idx],
        labels=restricted.labels[idx],
        split=restricted.split[idx],
        domain=restricted.domain[idx],
        meta={**restricted.meta, "k": k, "subset_seed": subset_seed,
              "class_subset": subset.tolist()},
    )
    return sub, shortfalls


def split_target_labeled_and_eval(
    tgt_logits: np.ndarray, tgt_labels: np.ndarray, k: int, seed: int,
    eval_frac: float, class_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint, per-class-stratified ``(labeled_cal_idx, eval_idx)`` split of the restricted
    TARGET_TEST rows."""
    rng = np.random.default_rng((seed, 7_000_000_000))
    cal_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []
    for c in range(k):
        rng_c = rng if class_ids is None else np.random.default_rng(
            (seed, NESTED_TARGET_TAG, int(class_ids[c]) + 1))
        idx_c = rng_c.permutation(np.where(tgt_labels == c)[0])
        if len(idx_c) == 0:
            continue
        n_eval = max(1, int(round(len(idx_c) * eval_frac)))
        n_eval = min(n_eval, len(idx_c))
        eval_parts.append(idx_c[:n_eval])
        cal_parts.append(idx_c[n_eval:])
    eval_idx = np.concatenate(eval_parts) if eval_parts else np.zeros(0, dtype=int)
    cal_idx = np.concatenate(cal_parts) if cal_parts else np.zeros(0, dtype=int)
    return cal_idx, eval_idx


def _build_eval_only_dataset(sub: ShiftDataset, eval_idx: np.ndarray) -> ShiftDataset:
    src_positions = np.where(sub.split == SOURCE_CAL)[0]
    tgt_positions = np.where(sub.split == TARGET_TEST)[0][eval_idx]
    keep = np.sort(np.concatenate([src_positions, tgt_positions]))
    return ShiftDataset(
        logits=sub.logits[keep], labels=sub.labels[keep],
        split=sub.split[keep], domain=sub.domain[keep], meta=dict(sub.meta),
    )


def run_acp_or_oracle(
    method: str, src_logits: np.ndarray, src_labels: np.ndarray,
    cal_pool_logits: np.ndarray, cal_pool_labels: np.ndarray,
    eval_logits: np.ndarray, eval_labels: np.ndarray, k: int, alpha: float,
) -> dict:
    """Fit ``acp``/``oracle`` directly on the disjoint labeled target cal-pool and score on the
    held-out eval set."""
    fitted = METHODS[method]().fit(
        src_logits, src_labels,
        target_labeled=(cal_pool_logits, cal_pool_labels),
        alpha=alpha, seed=0,
    )
    mask = fitted.predict_set(eval_logits)
    n_cal_k = calibration_counts_per_class(method, fitted, src_labels, cal_pool_labels, k)
    return {
        "marginal": marginal_coverage(mask, eval_labels),
        "worst_class": worst_class_coverage(mask, eval_labels, k),
        "frac_undercovered": frac_classes_undercovered(mask, eval_labels, k, 1.0 - alpha),
        "mean_set_size": mean_set_size(mask),
        "frac_sig_undercovered": frac_significantly_undercovered(
            mask, eval_labels, k, n_cal_k, alpha=alpha, q=SIG_Q),
        "per_class_coverage": _coverage_tuple(per_class_coverage(mask, eval_labels, k)),
        "n_cal_per_class": [int(n) for n in n_cal_k],
    }


def _coverage_tuple(values) -> tuple[float | None, ...]:
    return tuple(
        None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        for v in values
    )


def low_count_eval_classes(
    eval_labels: np.ndarray, k: int, floor: int = EVAL_COUNT_FLOOR,
) -> tuple[int, ...]:
    """Class ids with fewer than ``floor`` held-out eval examples."""
    counts = np.array([int((eval_labels == c).sum()) for c in range(k)])
    return tuple(int(c) for c in np.where(counts < floor)[0])


def _log_low_count_eval_classes(
    k: int, subset_seed: int, eval_labels: np.ndarray,
    low_count: tuple[int, ...],
) -> None:
    for c in low_count:
        n = int((eval_labels == c).sum())
        print(f"WARNING: K={k} seed={subset_seed} class {c}: only {n} "
              f"target eval examples (< {EVAL_COUNT_FLOOR} floor) -- "
              f"worst_class for this (K,seed) may be optimistic.")


def _log_shortfalls(k: int, subset_seed: int, n_cal_per_class: int,
                     shortfalls: dict[int, int]) -> None:
    for c, deficit in sorted(shortfalls.items()):
        available = n_cal_per_class - deficit
        print(f"WARNING: K={k} seed={subset_seed} class {c}: only "
              f"{available} source cal examples available "
              f"(< {n_cal_per_class} requested, short by {deficit}).")


def _run_one_method(
    method: str, src_logits: np.ndarray, src_labels: np.ndarray,
    cal_pool_logits: np.ndarray, cal_pool_labels: np.ndarray,
    eval_logits: np.ndarray, eval_labels: np.ndarray, sub_eval: ShiftDataset,
    k: int, alpha: float,
) -> tuple[dict, int]:
    if method in LABELED_TARGET_METHODS:
        row = run_acp_or_oracle(
            method, src_logits, src_labels, cal_pool_logits, cal_pool_labels,
            eval_logits, eval_labels, k, alpha,
        )
        return row, int(len(cal_pool_labels))
    row = run(sub_eval, method, alpha=alpha, seeds=1, eval_frac=1.0)
    row["per_class_coverage"] = _coverage_tuple(row["per_class_coverage_per_seed"][0])
    row["n_cal_per_class"] = [int(n) for n in row["n_cal_per_class_per_seed"][0]]
    return row, 0


def new_extras() -> dict:
    """Collector for the facts only the extended payload stores: one entry per sub-problem, and
    the per-class calibration counts of every cell."""
    return {"problems": [], "n_cal_per_class": {}}


def _problem_facts(k: int, subset_seed: int, class_subset: list[int],
                   eval_logits: np.ndarray, eval_labels: np.ndarray) -> dict:
    hit = eval_logits.argmax(axis=1) == eval_labels
    n_eval = [int((eval_labels == c).sum()) for c in range(k)]
    top1 = [float(hit[eval_labels == c].mean()) if n else None
            for c, n in zip(range(k), n_eval)]
    return {"k": k, "subset_seed": subset_seed, "class_ids": [int(c) for c in class_subset],
            "n_eval_per_class": n_eval, "eval_top1_per_class": top1}


def run_one_subproblem(
    ds: ShiftDataset, k: int, subset_seed: int, n_cal_per_class: int,
    alpha: float, methods: list[str], eval_frac: float = DEFAULT_EVAL_FRAC,
    design: str = "independent", extras: dict | None = None,
) -> list[SweepRecord]:
    """Restrict to one (K, subset_seed) sub-problem and run every method in ``methods``, every
    one scored on the SAME held-out target eval set: label-free methods
    (split/mondrian/weighted/ppi/clustered) run via the existing ``pcpbench.runner.run`` on an
    eval-only view of the restricted dataset; acp/oracle (``LABELED_TARGET_METHODS``) are fit
    directly (``run_acp_or_oracle``) on a DISJOINT labeled target cal-pool and scored on that
    same eval set."""
    sub, shortfalls = build_k_subproblem(ds, k, subset_seed, n_cal_per_class, design)
    _log_shortfalls(k, subset_seed, n_cal_per_class, shortfalls)
    n_cal_total = int((sub.split == SOURCE_CAL).sum())

    tgt_logits, tgt_labels = sub.target_test()
    cal_idx, eval_idx = split_target_labeled_and_eval(
        tgt_logits, tgt_labels, k, subset_seed, eval_frac,
        class_ids=np.asarray(sub.meta["class_subset"]) if design == "nested" else None,
    )
    eval_logits, eval_labels = tgt_logits[eval_idx], tgt_labels[eval_idx]
    cal_pool_logits, cal_pool_labels = tgt_logits[cal_idx], tgt_labels[cal_idx]

    low_count = low_count_eval_classes(eval_labels, k)
    _log_low_count_eval_classes(k, subset_seed, eval_labels, low_count)

    src_logits, src_labels = sub.source_cal()
    sub_eval = _build_eval_only_dataset(sub, eval_idx)
    if extras is not None:
        extras["problems"].append(_problem_facts(
            k, subset_seed, sub.meta["class_subset"], eval_logits, eval_labels))

    records = []
    for method in methods:
        row, n_target_cal_pool = _run_one_method(
            method, src_logits, src_labels, cal_pool_logits, cal_pool_labels,
            eval_logits, eval_labels, sub_eval, k, alpha,
        )
        if extras is not None:
            extras["n_cal_per_class"][(k, subset_seed, method)] = row["n_cal_per_class"]
        records.append(SweepRecord(
            k=k, subset_seed=subset_seed, method=method,
            marginal=row["marginal"], worst_class=row["worst_class"],
            frac_undercovered=row["frac_undercovered"],
            mean_set_size=row["mean_set_size"],
            set_over_k=row["mean_set_size"] / k,
            n_cal_total=n_cal_total, n_shortfall_classes=len(shortfalls),
            n_eval_total=int(len(eval_labels)), n_target_cal_pool=n_target_cal_pool,
            n_low_count_eval_classes=len(low_count),
            frac_sig_undercovered=float(row["frac_sig_undercovered"]),
            per_class_coverage=row["per_class_coverage"],
        ))
    return records


def run_sweep(
    ds: ShiftDataset, k_values: list[int], subset_seeds: int,
    n_cal_per_class: int, alpha: float, methods: list[str] | None = None,
    eval_frac: float = DEFAULT_EVAL_FRAC, design: str = "independent",
    extras: dict | None = None,
) -> list[SweepRecord]:
    """Run the full controlled sweep: every K in ``k_values`` x every subset seed in
    ``range(subset_seeds)`` x every method."""
    if design not in SUBSET_DESIGNS:
        raise ValueError(f"unknown subset design {design!r}; expected one of {SUBSET_DESIGNS}.")
    resolved_methods = methods if methods is not None else sorted(METHODS)
    records: list[SweepRecord] = []
    for k in k_values:
        for subset_seed in range(subset_seeds):
            records += run_one_subproblem(
                ds, k, subset_seed, n_cal_per_class, alpha, resolved_methods,
                eval_frac, design, extras,
            )
    return records


def no_shift_view(
    ds: ShiftDataset, view_seed: int,
    n_target_per_class: int = NO_SHIFT_TARGET_PER_CLASS,
) -> ShiftDataset:
    """A view of ``ds`` with no shift: ``n_target_per_class`` source-calibration rows of every
    class stand in for the target and the other source rows stay source calibration."""
    source_positions = np.where(ds.split == SOURCE_CAL)[0]
    source_labels = ds.labels[source_positions]
    target_parts: list[np.ndarray] = []
    for g in range(ds.n_classes):
        rows_g = source_positions[source_labels == g]
        if len(rows_g) <= n_target_per_class:
            raise ValueError(
                f"class {g} has {len(rows_g)} source calibration rows; the no-shift "
                f"view needs more than {n_target_per_class} to leave a source side.")
        rng = np.random.default_rng((view_seed, NO_SHIFT_VIEW_TAG, g + 1))
        target_parts.append(rng.permutation(rows_g)[:n_target_per_class])
    target_rows = np.concatenate(target_parts)
    split = np.where(np.isin(source_positions, target_rows), TARGET_TEST, SOURCE_CAL)
    return ShiftDataset(
        logits=ds.logits[source_positions],
        labels=source_labels,
        split=split.astype(ds.split.dtype),
        domain=ds.domain[source_positions],
        meta={**ds.meta, "view": "no_shift", "view_seed": view_seed,
              "n_target_per_class": n_target_per_class},
    )


def summarize(records: list[SweepRecord]) -> list[SweepSummary]:
    """Average +/- std over the subset seeds for each (K, method) pair."""
    groups: dict[tuple[int, str], list[SweepRecord]] = defaultdict(list)
    for r in records:
        groups[(r.k, r.method)].append(r)

    summaries = []
    for (k, method), rs in sorted(groups.items()):
        worst = np.array([r.worst_class for r in rs], dtype=float)
        marginal = np.array([r.marginal for r in rs], dtype=float)
        set_over_k = np.array([r.set_over_k for r in rs], dtype=float)
        sig = np.array([r.frac_sig_undercovered for r in rs], dtype=float)
        sig_all_nan = bool(np.all(np.isnan(sig)))
        summaries.append(SweepSummary(
            k=k, method=method, n_seeds=len(rs),
            worst_class_mean=float(worst.mean()), worst_class_std=float(worst.std()),
            marginal_mean=float(marginal.mean()), marginal_std=float(marginal.std()),
            set_over_k_mean=float(set_over_k.mean()), set_over_k_std=float(set_over_k.std()),
            frac_sig_undercovered_mean=float("nan") if sig_all_nan else float(np.nanmean(sig)),
            frac_sig_undercovered_std=float("nan") if sig_all_nan else float(np.nanstd(sig)),
        ))
    return summaries


def render_markdown(summaries: list[SweepSummary], k_values: list[int]) -> str:
    """Markdown table: mean +/- std worst-class coverage vs K, one row per method, one column per
    requested K."""
    by_method: dict[str, dict[int, SweepSummary]] = defaultdict(dict)
    for s in summaries:
        by_method[s.method][s.k] = s

    header = "| method | " + " | ".join(f"K={k}" for k in k_values) + " |"
    sep = "| --- |" + " --- |" * len(k_values)
    lines = ["## Controlled K-sweep: worst-class coverage vs K", "", header, sep]
    for method in sorted(by_method):
        cells = []
        for k in k_values:
            s = by_method[method].get(k)
            cells.append(f"{s.worst_class_mean:.3f} +/- {s.worst_class_std:.3f}" if s else "n/a")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_json_payload(
    cache_path: str, k_values: list[int], subset_seeds: int, n_cal_per_class: int,
    alpha: float, records: list[SweepRecord], summaries: list[SweepSummary],
    eval_frac: float = DEFAULT_EVAL_FRAC, acp_tie_break: str | None = None,
) -> dict:
    """The JSON payload of a sweep, with ``acp_tie_break``, the tie rule of ACP's audited
    probabilities, exactly when ACP ran."""
    config = {
        "cache": cache_path, "k_values": k_values, "subset_seeds": subset_seeds,
        "n_cal_per_class": n_cal_per_class, "alpha": alpha, "eval_frac": eval_frac,
    }
    if acp_tie_break is not None:
        config["acp_tie_break"] = acp_tie_break
    return {
        "config": config,
        "records": [asdict(r) for r in records],
        "summary": [s._asdict() for s in summaries],
    }


def source_cal_digest(ds: ShiftDataset) -> str:
    """SHA-256 of the source-calibration logits and labels."""
    logits, labels = ds.source_cal()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return digest.hexdigest()


def provenance(ds: ShiftDataset, stamp: str | None) -> dict:
    """What a reader needs to tell which code and which cache produced a file."""
    return {
        "code_version": stamp,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "source_cal_sha256": source_cal_digest(ds),
        "n_source_cal": int((ds.split == SOURCE_CAL).sum()),
        "n_target_test": int((ds.split == TARGET_TEST).sum()),
        "n_classes": ds.n_classes,
    }


def extend_payload(
    payload: dict, *, subset_design: str, view: str, view_seed: int | None,
    class_orders: list[list[int]] | None, provenance_block: dict, extras: dict,
) -> dict:
    """The default payload plus the keys of the nested design and the no-shift view."""
    config = {**payload["config"], "subset_design": subset_design, "view": view,
              "view_seed": view_seed, "method_seed": 0}
    counts = extras["n_cal_per_class"]
    records = [{**r, "n_cal_per_class": counts[(r["k"], r["subset_seed"], r["method"])]}
               for r in payload["records"]]
    return {**payload, "config": config, "records": records, "class_orders": class_orders,
            "problems": extras["problems"], "provenance": provenance_block}


def write_payload(payload: dict, out_path: Path) -> None:
    """Write the payload under a temporary name in the output directory and rename it into place,
    so a job killed mid-write leaves no file that a rerun would take for a finished one."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    tmp_path = out_path.with_name(f"{out_path.name}.tmp{os.getpid()}")
    if out_path.suffix == ".gz":
        with open(tmp_path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                zipped.write(text.encode("utf-8"))
    else:
        tmp_path.write_text(text)
    os.replace(tmp_path, out_path)


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="controlled_k_sweep",
        description=(
            "Controlled K-sweep: vary the number of classes K on ONE cache "
            "(random K-class subsets, per-class calibration size held "
            "constant) to isolate the worst-of-K effect from any "
            "dataset/classifier confound."
        ),
    )
    parser.add_argument(
        "--cache", required=True,
        help="Path to a pcpbench ShiftDataset .npz cache, or the literal 'synthetic'.",
    )
    parser.add_argument("--k-values", required=True, help="Comma-separated K values, e.g. 10,25,50,100.")
    parser.add_argument(
        "--subset-seeds", type=int, default=5,
        help="R: number of random-class-subset seeds averaged over per K (design default 5).",
    )
    parser.add_argument("--n-cal-per-class", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--eval-frac", type=float, default=DEFAULT_EVAL_FRAC,
        help=(
            "Fraction of each class's restricted target rows held out "
            "as the eval set every method is scored on; the remainder is the "
            "labeled cal-pool acp/oracle calibrate on (default 0.5)."
        ),
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--methods", default=None,
        help="Optional comma-separated method subset (default: all 6 -- split,mondrian,weighted,ppi,acp,oracle).",
    )
    parser.add_argument(
        "--subset-design", choices=SUBSET_DESIGNS, default="independent",
        help=(
            "independent (released): one independent class subset per (K, seed). "
            "nested: the K-subset is the first K classes of the seed's class order "
            "and every class keeps its rows at every K."
        ),
    )
    parser.add_argument(
        "--view", choices=VIEWS, default="shift",
        help=(
            "shift (released): the cache's real target. no_shift: "
            f"{NO_SHIFT_TARGET_PER_CLASS} source rows per class stand in for the target."
        ),
    )
    parser.add_argument(
        "--view-seed", type=int, default=0,
        help="Seed of the no-shift view's row draw (read only with --view no_shift).",
    )
    parser.add_argument(
        "--stamp", default=None,
        help=("Code version to record in the output of a nested or no-shift run, for "
              "example `git rev-parse --short HEAD`. A default run writes the released "
              "schema, which has no place for it."),
    )
    return parser


def _load_cache(path: str) -> ShiftDataset:
    if path == "synthetic":
        return make_synthetic_shift()
    return ShiftDataset.from_npz(path)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ds = _load_cache(args.cache)
    k_values = _parse_int_list(args.k_values)
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else sorted(METHODS)

    for k in k_values:
        if k > ds.n_classes:
            raise SystemExit(f"FATAL: requested K={k} exceeds cache n_classes={ds.n_classes}.")

    extended = args.subset_design != "independent" or args.view != "shift"
    provenance_block = provenance(ds, args.stamp) if extended else None
    extras = new_extras() if extended else None
    sweep_ds = ds
    if args.view == "no_shift":
        try:
            sweep_ds = no_shift_view(ds, args.view_seed)
        except ValueError as exc:
            raise SystemExit(f"FATAL: {exc}") from exc

    records = run_sweep(sweep_ds, k_values, args.subset_seeds, args.n_cal_per_class,
                        args.alpha, methods, args.eval_frac, args.subset_design, extras)
    summaries = summarize(records)

    md = render_markdown(summaries, k_values)
    print(md)

    payload = build_json_payload(
        args.cache, k_values, args.subset_seeds, args.n_cal_per_class, args.alpha, records,
        summaries, args.eval_frac,
        acp_tie_break=ACP_TIE_BREAK if "acp" in methods else None,
    )
    if extended:
        class_orders = None
        if args.subset_design == "nested":
            class_orders = [nested_class_order(sweep_ds.n_classes, seed).tolist()
                            for seed in range(args.subset_seeds)]
        payload = extend_payload(
            payload, subset_design=args.subset_design, view=args.view,
            view_seed=args.view_seed if args.view == "no_shift" else None,
            class_orders=class_orders, provenance_block=provenance_block, extras=extras,
        )
    out_path = Path(args.out)
    write_payload(payload, out_path)
    print(f"wrote {len(records)} records, {len(summaries)} summary rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
