"""Label-complexity m-sweep driver: worst_class(m) per method, disjoint eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pcpbench.calibrators.acp import DEFAULT_TIE_BREAK as ACP_TIE_BREAK
from pcpbench.datasets.shift_dataset import ShiftDataset
from pcpbench.metrics.coverage import (
    frac_classes_undercovered,
    marginal_coverage,
    mean_set_size,
    per_class_coverage,
    worst_class_coverage,
)
from pcpbench.metrics.validity_floor import frac_significantly_undercovered
from pcpbench.runner import SIG_Q, METHODS, calibration_counts_per_class

LABEL_BUDGET_METHODS = frozenset({"oracle", "ppi", "acp"})
TARGET_LOGITS_METHODS = frozenset({"weighted", "ppi"})
EVAL_COUNT_FLOOR = 5


class SweepRow(NamedTuple):
    """One (method, m, seed) evaluation: the atomic unit of the sweep."""

    method: str
    m: int
    seed: int
    n_labeled: int
    per_class_labeled: tuple[int, ...]
    n_shortfall_classes: int
    worst_class: float
    marginal: float
    label_budget_method: bool
    mean_set_size: float
    set_over_k: float
    frac_undercovered: float
    frac_sig_undercovered: float
    per_class_coverage: tuple[float | None, ...]


class SweepSummary(NamedTuple):
    """Mean +/- std over seeds for one (method, m) cell."""

    method: str
    m: int
    n_labeled_mean: float
    n_shortfall_classes_mean: float
    worst_class_mean: float
    worst_class_std: float
    marginal_mean: float
    marginal_std: float
    label_budget_method: bool
    note: str
    mean_set_size_mean: float
    mean_set_size_std: float
    set_over_k_mean: float
    set_over_k_std: float
    frac_undercovered_mean: float
    frac_undercovered_std: float
    frac_sig_undercovered_mean: float
    frac_sig_undercovered_std: float


class EvalClassReport(NamedTuple):
    """Per-seed audit of the held-out eval set's per-class example counts."""

    seed: int
    counts: tuple[int, ...]
    low_count_classes: tuple[int, ...]


def split_pool_and_eval(
    tgt_logits: np.ndarray, tgt_labels: np.ndarray, K: int, seed: int,
    eval_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint, per-class-stratified (pool_idx, eval_idx) partition of target-test."""
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


_BUDGET_PERM_STREAM = 3_000_000_000


def sample_budget(
    pool_logits: np.ndarray, pool_labels: np.ndarray, m: int, K: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the first ``m`` per class from a SEEDED, m-INDEPENDENT per-class permutation of the
    budget pool, so the budgets of one seed are nested."""
    rng = np.random.default_rng((seed, _BUDGET_PERM_STREAM))
    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    for c in range(K):
        idx_c = np.where(pool_labels == c)[0]
        if len(idx_c) == 0:
            continue
        perm_c = rng.permutation(idx_c)
        take = min(m, len(perm_c))
        chosen = perm_c[:take]
        parts_x.append(pool_logits[chosen])
        parts_y.append(pool_labels[chosen])
    if not parts_x:
        return pool_logits[:0], pool_labels[:0]
    return np.concatenate(parts_x), np.concatenate(parts_y)


def per_class_label_counts(labels: np.ndarray, K: int) -> tuple[int, ...]:
    """Count of ``labels`` per class 0..K-1, class by class rather than in aggregate."""
    labels = np.asarray(labels)
    return tuple(int((labels == c).sum()) for c in range(K))


def _fit_one(
    method: str, src_logits: np.ndarray, src_labels: np.ndarray,
    full_tgt_logits: np.ndarray, budget: tuple[np.ndarray, np.ndarray] | None,
    alpha: float, seed: int,
):
    kwargs: dict = {"alpha": alpha, "seed": seed}
    if method in TARGET_LOGITS_METHODS:
        kwargs["target_logits"] = full_tgt_logits
    if method in LABEL_BUDGET_METHODS and budget is not None:
        kwargs["target_labeled"] = budget
    return METHODS[method]().fit(src_logits, src_labels, **kwargs)


def _log_capacity_capped(method: str, m: int, seed: int, per_class: tuple[int, ...]) -> None:
    short = {c: n for c, n in enumerate(per_class) if n < m}
    for c, n in sorted(short.items()):
        print(f"WARNING: method={method} m={m} seed={seed} class {c}: only "
              f"{n} labeled target examples available in the budget pool "
              f"(< {m} requested) -- capacity-capped, NOT the full m.")


def _evaluate_one(
    method: str, m: int, seed: int,
    src_logits: np.ndarray, src_labels: np.ndarray, full_tgt_logits: np.ndarray,
    pool_logits: np.ndarray, pool_labels: np.ndarray,
    eval_logits: np.ndarray, eval_labels: np.ndarray,
    K: int, alpha: float, label_budget_method: bool,
) -> SweepRow:
    if label_budget_method:
        budget_logits, budget_labels = sample_budget(pool_logits, pool_labels, m, K, seed)
        n_labeled = int(len(budget_labels))
        per_class = per_class_label_counts(budget_labels, K)
        n_shortfall_classes = sum(1 for n in per_class if n < m)
        if n_shortfall_classes:
            _log_capacity_capped(method, m, seed, per_class)
        budget = (budget_logits, budget_labels) if n_labeled > 0 else None
    else:
        n_labeled = 0
        per_class = tuple(0 for _ in range(K))
        n_shortfall_classes = 0
        budget = None

    try:
        fitted = _fit_one(method, src_logits, src_labels, full_tgt_logits, budget, alpha, seed)
    except ValueError:
        nan = float("nan")
        return SweepRow(method, m, seed, n_labeled, per_class, n_shortfall_classes,
                        nan, nan, label_budget_method,
                        mean_set_size=nan, set_over_k=nan, frac_undercovered=nan,
                        frac_sig_undercovered=nan,
                        per_class_coverage=tuple(None for _ in range(K)))

    mask = fitted.predict_set(eval_logits)
    set_size = mean_set_size(mask)
    n_cal_k = calibration_counts_per_class(
        method, fitted, src_labels, None if budget is None else budget[1], K
    )
    return SweepRow(
        method=method, m=m, seed=seed, n_labeled=n_labeled,
        per_class_labeled=per_class, n_shortfall_classes=n_shortfall_classes,
        worst_class=worst_class_coverage(mask, eval_labels, K),
        marginal=marginal_coverage(mask, eval_labels),
        label_budget_method=label_budget_method,
        mean_set_size=set_size,
        set_over_k=set_size / K,
        frac_undercovered=frac_classes_undercovered(mask, eval_labels, K, 1.0 - alpha),
        frac_sig_undercovered=frac_significantly_undercovered(
            mask, eval_labels, K, n_cal_k, alpha=alpha, q=SIG_Q),
        per_class_coverage=tuple(
            None if np.isnan(v) else float(v)
            for v in per_class_coverage(mask, eval_labels, K)
        ),
    )


def run_sweep(
    ds: ShiftDataset, methods: list[str], m_values: list[int],
    alpha: float, seeds: int, eval_frac: float = 0.5,
) -> list[SweepRow]:
    """Run the full (method, m, seed) grid and return every atomic row."""
    src_logits, src_labels = ds.source_cal()
    full_tgt_logits, full_tgt_labels = ds.target_test()
    K = ds.n_classes
    rows: list[SweepRow] = []
    for seed in range(seeds):
        pool_idx, eval_idx = split_pool_and_eval(full_tgt_logits, full_tgt_labels, K, seed,
                                                 eval_frac)
        pool_logits, pool_labels = full_tgt_logits[pool_idx], full_tgt_labels[pool_idx]
        eval_logits, eval_labels = full_tgt_logits[eval_idx], full_tgt_labels[eval_idx]
        for method in methods:
            label_budget_method = method in LABEL_BUDGET_METHODS
            for m in m_values:
                rows.append(_evaluate_one(
                    method, m, seed, src_logits, src_labels, full_tgt_logits,
                    pool_logits, pool_labels, eval_logits, eval_labels,
                    K, alpha, label_budget_method,
                ))
    return rows


def _note_for(label_budget_method: bool, worst: np.ndarray, m: int,
              n_shortfall_mean: float) -> str:
    if not label_budget_method:
        return "budget-independent: method does not consume a labeled-target budget"
    if np.all(np.isnan(worst)):
        return "undefined at this m: zero labeled target points (method requires target_labeled)"
    if n_shortfall_mean > 0:
        return (f"capacity-capped: avg {n_shortfall_mean:.1f} class(es)/seed had "
                f"fewer than m={m} labels available in the budget pool (see "
                f"per_seed[*].per_class_labeled for exact per-class counts)")
    return ""


def _nan_mean_std(values: np.ndarray) -> tuple[float, float]:
    if np.all(np.isnan(values)):
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values))


def summarize(rows: list[SweepRow]) -> list[SweepSummary]:
    """Aggregate ``rows`` to one mean+std ``SweepSummary`` per (method, m) cell."""
    keys = sorted({(r.method, r.m) for r in rows})
    out: list[SweepSummary] = []
    for method, m in keys:
        group = [r for r in rows if r.method == method and r.m == m]
        worst = np.array([r.worst_class for r in group], dtype=float)
        marg = np.array([r.marginal for r in group], dtype=float)
        n_lab = np.array([r.n_labeled for r in group], dtype=float)
        n_short = np.array([r.n_shortfall_classes for r in group], dtype=float)
        label_budget_method = group[0].label_budget_method
        n_shortfall_mean = float(np.mean(n_short))
        added = {
            name: _nan_mean_std(np.array([getattr(r, name) for r in group], dtype=float))
            for name in ("mean_set_size", "set_over_k", "frac_undercovered",
                         "frac_sig_undercovered")
        }
        out.append(SweepSummary(
            method=method, m=m,
            n_labeled_mean=float(np.mean(n_lab)),
            n_shortfall_classes_mean=n_shortfall_mean,
            worst_class_mean=float(np.nanmean(worst)) if not np.all(np.isnan(worst)) else float("nan"),
            worst_class_std=float(np.nanstd(worst)) if not np.all(np.isnan(worst)) else float("nan"),
            marginal_mean=float(np.nanmean(marg)) if not np.all(np.isnan(marg)) else float("nan"),
            marginal_std=float(np.nanstd(marg)) if not np.all(np.isnan(marg)) else float("nan"),
            label_budget_method=label_budget_method,
            note=_note_for(label_budget_method, worst, m, n_shortfall_mean),
            mean_set_size_mean=added["mean_set_size"][0],
            mean_set_size_std=added["mean_set_size"][1],
            set_over_k_mean=added["set_over_k"][0],
            set_over_k_std=added["set_over_k"][1],
            frac_undercovered_mean=added["frac_undercovered"][0],
            frac_undercovered_std=added["frac_undercovered"][1],
            frac_sig_undercovered_mean=added["frac_sig_undercovered"][0],
            frac_sig_undercovered_std=added["frac_sig_undercovered"][1],
        ))
    return out


def format_markdown(summary: list[SweepSummary]) -> str:
    """Render one worst_class-vs-m markdown table per method."""
    methods = sorted({s.method for s in summary})
    lines: list[str] = []
    for method in methods:
        method_rows = sorted((s for s in summary if s.method == method), key=lambda s: s.m)
        lines.append(f"### {method}")
        lines.append("| m | n_labeled (actual) | n_shortfall_classes | worst_class | marginal | note |")
        lines.append("|---|---|---|---|---|---|")
        for s in method_rows:
            wc = "n/a" if np.isnan(s.worst_class_mean) else f"{s.worst_class_mean:.3f} +/- {s.worst_class_std:.3f}"
            mg = "n/a" if np.isnan(s.marginal_mean) else f"{s.marginal_mean:.3f} +/- {s.marginal_std:.3f}"
            lines.append(f"| {s.m} | {s.n_labeled_mean:.1f} | {s.n_shortfall_classes_mean:.1f} "
                         f"| {wc} | {mg} | {s.note} |")
        lines.append("")
    return "\n".join(lines)


def _log_low_count_eval_classes(report: "EvalClassReport") -> None:
    if not report.low_count_classes:
        return
    detail = {c: report.counts[c] for c in report.low_count_classes}
    print(f"WARNING: seed={report.seed}: eval classes with < {EVAL_COUNT_FLOOR} "
          f"examples: {detail}")


def build_eval_class_report(
    ds: ShiftDataset, seeds: int, eval_frac: float, floor: int = EVAL_COUNT_FLOOR,
) -> list[EvalClassReport]:
    """Per-seed classwise eval-example counts + which classes fall below ``floor``."""
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    reports: list[EvalClassReport] = []
    for seed in range(seeds):
        _, eval_idx = split_pool_and_eval(tgt_logits, tgt_labels, K, seed, eval_frac)
        eval_labels = tgt_labels[eval_idx]
        counts = per_class_label_counts(eval_labels, K)
        low = tuple(c for c in range(K) if counts[c] < floor)
        report = EvalClassReport(seed=seed, counts=counts, low_count_classes=low)
        _log_low_count_eval_classes(report)
        reports.append(report)
    return reports


def _json_safe(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def to_json_payload(
    rows: list[SweepRow], summary: list[SweepSummary], *,
    cache: str, alpha: float, seeds: int, m_values: list[int], methods: list[str],
    eval_frac: float, eval_class_report: list[EvalClassReport] = (),
) -> dict:
    """The JSON payload of the sweep; a run that includes ACP records its tie rule as
    ``acp_tie_break``."""
    payload = {
        "cache": cache, "alpha": alpha, "seeds": seeds,
        "m_values": m_values, "methods": methods, "eval_frac": eval_frac,
        "per_seed": [{k: _json_safe(v) for k, v in row._asdict().items()} for row in rows],
        "summary": [{k: _json_safe(v) for k, v in s._asdict().items()} for s in summary],
        "eval_class_report": [r._asdict() for r in eval_class_report],
    }
    if "acp" in methods:
        payload["acp_tie_break"] = ACP_TIE_BREAK
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label-complexity m-sweep driver.")
    p.add_argument("--cache", required=True, type=Path,
                    help="Path to a ShiftDataset .npz cache (5-key: logits/labels/split/domain/meta).")
    p.add_argument("--m-values", required=True, type=str,
                    help="Comma-separated labeled-target-budget-per-class values, e.g. 0,10,25,50,100,200.")
    p.add_argument("--methods", required=True, type=str,
                    help="Comma-separated method names from pcpbench.runner.METHODS, e.g. ppi,acp,oracle.")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--eval-frac", type=float, default=0.5,
                    help="Fraction of each class's target-test points held out as the eval set.")
    p.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    m_values = [int(x) for x in args.m_values.split(",")]
    methods = [x.strip() for x in args.methods.split(",")]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; choose from {sorted(METHODS)}")

    ds = ShiftDataset.from_npz(args.cache)
    rows = run_sweep(ds, methods, m_values, args.alpha, args.seeds, args.eval_frac)
    summary = summarize(rows)
    eval_class_report = build_eval_class_report(ds, args.seeds, args.eval_frac)
    payload = to_json_payload(
        rows, summary, cache=str(args.cache), alpha=args.alpha, seeds=args.seeds,
        m_values=m_values, methods=methods, eval_frac=args.eval_frac,
        eval_class_report=eval_class_report,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(format_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
