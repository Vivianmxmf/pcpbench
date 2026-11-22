#!/usr/bin/env python3
"""No-shift reference: measure every method against the validity floor when no shift is present
at all."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from pcpbench.datasets.shift_dataset import SOURCE_CAL, SOURCE_TRAIN, TARGET_TEST, ShiftDataset
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.validity_floor import FloorEstimate, expected_min_coverage
from pcpbench.runner import METHODS, run

_AB_SEED_SENTINEL = 3_000_000_000

_METRIC_KEYS = ("marginal", "worst_class", "frac_undercovered",
                "frac_sig_undercovered", "mean_set_size")


def split_calibration_halves(
    src_logits: np.ndarray, src_labels: np.ndarray, K: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint, per-class-stratified halves ``(a_idx, b_idx)`` of the source calibration
    partition (positions into ``src_logits``/``src_labels``, i.e. into an already-filtered
    ``ShiftDataset.source_cal()`` view)."""
    rng = np.random.default_rng((seed, _AB_SEED_SENTINEL))
    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    for c in range(K):
        idx_c = rng.permutation(np.where(src_labels == c)[0])
        if len(idx_c) == 0:
            continue
        n_a = len(idx_c) // 2
        a_parts.append(idx_c[:n_a])
        b_parts.append(idx_c[n_a:])
    a_idx = np.concatenate(a_parts) if a_parts else np.zeros(0, dtype=int)
    b_idx = np.concatenate(b_parts) if b_parts else np.zeros(0, dtype=int)
    return a_idx, b_idx


def build_no_shift_view(
    ds: ShiftDataset, a_idx: np.ndarray, b_idx: np.ndarray,
) -> ShiftDataset:
    """A view of ``ds`` whose source-calibration partition is split into two halves: A keeps the
    source-calibration split code, B takes the target-test code and stands in for the whole
    target-test set."""
    src_positions = np.where(ds.split == SOURCE_CAL)[0]
    train_positions = np.where(ds.split == SOURCE_TRAIN)[0]
    a_positions = src_positions[a_idx]
    b_positions = src_positions[b_idx]

    keep = np.concatenate([train_positions, a_positions, b_positions])
    new_split = np.concatenate([
        ds.split[train_positions],
        np.full(len(a_positions), SOURCE_CAL, dtype=ds.split.dtype),
        np.full(len(b_positions), TARGET_TEST, dtype=ds.split.dtype),
    ])
    return ShiftDataset(
        logits=ds.logits[keep],
        labels=ds.labels[keep],
        split=new_split,
        domain=ds.domain[keep],
        meta=dict(ds.meta),
    )


def _nan_reduce(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _floor_estimate_for_row(row: dict, K: int, alpha: float, view_seed: int):
    n_cal = row["n_cal_per_class_per_seed"][0]
    n_eval = row["n_eval_per_class_per_seed"][0]
    if n_cal is None:
        return None
    cal = np.asarray(n_cal, dtype=np.int64)
    ev = np.asarray(n_eval, dtype=np.int64)
    keep = (cal > 0) & (ev > 0)
    if not keep.any():
        return FloorEstimate(
            K=0, n_cal=None, n_eval=None, alpha=float(alpha), reps=0,
            mean=1.0, sd=0.0, q05=1.0, q95=1.0,
        )
    return expected_min_coverage(
        int(keep.sum()), cal[keep], ev[keep], alpha, seed=view_seed,
    )


def _floor_class_counts(row: dict) -> tuple[int, int, int] | None:
    n_cal = row["n_cal_per_class_per_seed"][0]
    if n_cal is None:
        return None
    cal = np.asarray(n_cal, dtype=np.int64)
    ev = np.asarray(row["n_eval_per_class_per_seed"][0], dtype=np.int64)
    no_eval = ev == 0
    no_cal = (ev > 0) & (cal == 0)
    return int((~no_eval & ~no_cal).sum()), int(no_eval.sum()), int(no_cal.sum())


def _aggregate_floor(estimates: list) -> tuple[float, float, str, int]:
    ok = [e for e in estimates if e is not None]
    if not ok:
        return float("nan"), float("nan"), "no_successful_seed", 0
    floor_mean = float(np.mean([e.mean for e in ok]))
    floor_sd = float(np.mean([e.sd for e in ok]))
    status = "zero_sd" if floor_sd == 0.0 else "ok"
    return floor_mean, floor_sd, status, len(ok)


def _json_safe(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def build_payload(
    ds: ShiftDataset, methods: list[str], alpha: float, seeds: int,
    eval_frac: float, cache_path: str,
) -> dict:
    """Run the no-shift reference for every ``method`` over ``seeds`` view seeds and assemble the
    ``{cache, alpha, seeds, eval_frac, per_seed, summary, floor}`` payload."""
    K = ds.n_classes
    src_logits, src_labels = ds.source_cal()

    per_seed_records: list[dict] = []
    rows_by_method: dict[str, list[dict]] = {m: [] for m in methods}
    floor_estimates_by_method: dict[str, list] = {m: [] for m in methods}
    floor_counts_by_method: dict[str, list] = {m: [] for m in methods}

    for method in methods:
        for s in range(seeds):
            a_idx, b_idx = split_calibration_halves(src_logits, src_labels, K, s)
            view = build_no_shift_view(ds, a_idx, b_idx)
            row = run(view, method, alpha, seeds=1, m_labeled=0, eval_frac=eval_frac)
            seed_floor = _floor_estimate_for_row(row, K, alpha, s)
            counts = _floor_class_counts(row)
            per_seed_records.append({
                "method": method,
                "view_seed": s,
                "row": row,
                "floor_mean": seed_floor.mean if seed_floor is not None else float("nan"),
                "floor_sd": seed_floor.sd if seed_floor is not None else float("nan"),
                "floor_n_classes": None if counts is None else counts[0],
                "floor_n_no_eval_classes": None if counts is None else counts[1],
                "floor_n_no_cal_classes": None if counts is None else counts[2],
            })
            rows_by_method[method].append(row)
            floor_estimates_by_method[method].append(seed_floor)
            floor_counts_by_method[method].append(counts)

    floor: dict[str, dict] = {}
    floor_agg: dict[str, tuple] = {}
    for method in methods:
        floor_mean_v, floor_sd_v, status, n_ok = _aggregate_floor(
            floor_estimates_by_method[method]
        )
        floor_agg[method] = (floor_mean_v, floor_sd_v, status, n_ok)
        ok_counts = [c for c in floor_counts_by_method[method] if c is not None]
        floor[method] = {
            "mean": floor_mean_v,
            "sd": floor_sd_v,
            "status": status,
            "n_successful_seeds": n_ok,
            "max_no_eval_classes": max((c[1] for c in ok_counts), default=None),
            "max_no_cal_classes": max((c[2] for c in ok_counts), default=None),
        }

    summary: list[dict] = []
    for method in methods:
        rows = rows_by_method[method]
        entry: dict = {"method": method, "n_seeds": int(seeds)}
        for key in _METRIC_KEYS:
            mean, std = _nan_reduce([r[key] for r in rows])
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
        floor_mean_v, floor_sd_v, status, n_ok = floor_agg[method]
        if status == "ok":
            excess = entry["worst_class_mean"] - floor_mean_v
            z = excess / floor_sd_v
        elif status == "zero_sd":
            excess = entry["worst_class_mean"] - floor_mean_v
            z = float("nan")
        else:
            excess = float("nan")
            z = float("nan")
        entry["floor_mean"] = floor_mean_v
        entry["floor_sd"] = floor_sd_v
        entry["excess"] = excess
        entry["z"] = z
        entry["floor_status"] = status
        entry["floor_n_seeds"] = n_ok
        summary.append(entry)

    return {
        "cache": cache_path,
        "alpha": float(alpha),
        "seeds": int(seeds),
        "eval_frac": float(eval_frac),
        "per_seed": per_seed_records,
        "summary": summary,
        "floor": floor,
    }


def _load_cache(path: str) -> ShiftDataset:
    if path == "synthetic":
        return make_synthetic_shift()
    return ShiftDataset.from_npz(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="no_shift_reference",
        description=(
            "No-shift reference: split a cache's source calibration partition "
            "into two halves and run every method on that no-shift view, so "
            "the validity floor's finite-sample noise can be told apart from "
            "shift-induced failure on the same cache."
        ),
    )
    parser.add_argument(
        "--cache", required=True,
        help="Path to a pcpbench ShiftDataset .npz cache, or the literal 'synthetic'.",
    )
    parser.add_argument(
        "--methods", default=None,
        help="Optional comma-separated method subset (default: every registry method).",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eval-frac", type=float, default=0.5)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ds = _load_cache(args.cache)
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else sorted(METHODS)

    payload = build_payload(ds, methods, args.alpha, args.seeds, args.eval_frac, args.cache)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.partial")
    try:
        tmp_path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"wrote no-shift reference for {len(methods)} method(s), {args.seeds} view "
          f"seeds -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
