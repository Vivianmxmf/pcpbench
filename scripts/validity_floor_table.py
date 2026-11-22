#!/usr/bin/env python3
"""Validity floor table: where the paper's oracle worst-class values sit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from pcpbench.metrics.validity_floor import (
    expected_min_coverage,
    seed_averaged_floor,
    significance_cutoff,
    undercovered_false_positive_rate,
)

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.1
LEVEL = 0.9
SIG_Q = 0.05
POWER_COVERAGE = 0.80
SEED_AVG_SEEDS = 10
SEED_AVG_SEEDS_KSWEEP = 5
K_SWEEP_CACHES = ("contrast_s1", "contrast_s2", "contrast_s3", "brightness_s1", "brightness_s3")
K_SWEEP_TYPED = {10: 0.798, 100: 0.724}
K_SWEEP_SOURCE = "the submitted paper, Section Controlled class-count decay"
LABEL_CACHE = "label_complexity_contrast_s3.json"
LABEL_TYPED = {10: 0.592, 25: 0.708, 50: 0.724}
IMAGENET_ORACLE = 0.559
IMAGENET_FRAC_UNDER = 0.306
IMAGENET_SOURCE = "the submitted paper, Section ImageNet-1k at K=1000"
IMAGENET_ROW_PREFIX = "imagenet1k_"
NUCLA_TARGET_COUNTS = (141, 300, 145, 148, 149)
NUCLA_ARTIFACT = "outputs/ntu2nucla_oracle.json"
EVAL_FRAC = 0.5
LEADERBOARD_ROW_SEEDS = 10
LEADERBOARD_ARTIFACT_ROWS = ("ntu60_xsub", "ntu2nucla")
SEED_AVG_REPS_DIVISOR = 4
POWER_N_EVAL = 50
POWER_CAL_RANGE = (50, 1000)


def _load(rel: str) -> Any:
    with open(ROOT / rel) as fh:
        return json.load(fh)


def k_sweep_oracle_means() -> dict[int, float]:
    per_k: dict[int, list[float]] = {}
    for cache in K_SWEEP_CACHES:
        data = _load(f"outputs/analysis/controlled_k_sweep_{cache}.json")
        for row in data["summary"]:
            if row["method"] == "oracle":
                per_k.setdefault(int(row["k"]), []).append(float(row["worst_class_mean"]))
    for k, vals in per_k.items():
        if len(vals) != len(K_SWEEP_CACHES):
            raise SystemExit(f"K={k}: expected {len(K_SWEEP_CACHES)} caches, found {len(vals)}")
    return {k: float(np.mean(v)) for k, v in per_k.items()}


def label_sweep_values(method: str) -> dict[int, float]:
    data = _load(f"outputs/analysis/{LABEL_CACHE}")
    return {
        int(r["m"]): float(r["worst_class_mean"])
        for r in data["summary"]
        if r["method"] == method and r["worst_class_mean"] is not None
    }


def paper_value_counts() -> dict[str, Any]:
    """Number of worst-class values each paper oracle value averages: per K for the K sweep
    (every oracle record over the five caches, five caches by five subset seeds), the label
    sweep's seed count, the valid seeds of each leaderboard artifact row, and for ImageNet the
    imagenet1k oracle rows times the seeds behind each leaderboard row."""
    ks: dict[int, int] = {}
    for cache in K_SWEEP_CACHES:
        data = _load(f"outputs/analysis/controlled_k_sweep_{cache}.json")
        for r in data["records"]:
            if r["method"] == "oracle":
                ks[int(r["k"])] = ks.get(int(r["k"]), 0) + 1
    lb = {name: int(_load(f"outputs/{name}_oracle.json")["n_valid_seeds"]) for name in LEADERBOARD_ARTIFACT_ROWS}
    if set(lb.values()) != {LEADERBOARD_ROW_SEEDS}:
        raise SystemExit(f"leaderboard artifacts record {lb} valid seeds, expected {LEADERBOARD_ROW_SEEDS}")
    return {
        "k_sweep": ks,
        "label_sweep": int(_load(f"outputs/analysis/{LABEL_CACHE}")["seeds"]),
        "leaderboard": lb,
        "imagenet1k": imagenet_oracle_values()["n_rows"] * LEADERBOARD_ROW_SEEDS,
    }


def leaderboard_oracle(dataset: str) -> float:
    with open(ROOT / "leaderboard.md") as fh:
        for line in fh:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and cells[0] == dataset and cells[1] == "oracle":
                return float(cells[3])
    raise SystemExit(f"leaderboard.md has no oracle row for {dataset}")


def nucla_sizes() -> tuple[np.ndarray, np.ndarray]:
    """(n_cal, n_eval) per class for ntu2nucla: n_eval = max(1, round(n * EVAL_FRAC)) with
    Python's round, n_cal = the remaining labeled pool."""
    ev = [min(n, max(1, int(round(n * EVAL_FRAC)))) for n in NUCLA_TARGET_COUNTS]
    cal = [n - e for n, e in zip(NUCLA_TARGET_COUNTS, ev)]
    art = _load(NUCLA_ARTIFACT)
    for name, arr, key in (("evaluation", ev, "n_eval"), ("pool", cal, "n_pool")):
        if sum(arr) != int(art[key]):
            raise SystemExit(
                f"ntu2nucla {name} sizes sum to {sum(arr)}, {NUCLA_ARTIFACT} records {key} {art[key]}"
            )
    return np.array(cal), np.array(ev)


def imagenet_oracle_values() -> dict[str, float]:
    """Mean worst_class and frac_undercovered over the leaderboard rows whose dataset starts with
    IMAGENET_ROW_PREFIX and whose method is oracle."""
    worst: list[float] = []
    frac: list[float] = []
    with open(ROOT / "leaderboard.md") as fh:
        for line in fh:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 5 and cells[0].startswith(IMAGENET_ROW_PREFIX) and cells[1] == "oracle":
                worst.append(float(cells[3]))
                frac.append(float(cells[4]))
    if not worst:
        raise SystemExit("leaderboard.md has no imagenet1k oracle rows")
    return {"n_rows": len(worst), "worst_class": float(np.mean(worst)), "frac_undercovered": float(np.mean(frac))}


def check_imagenet() -> dict[str, float]:
    got = imagenet_oracle_values()
    _check(f"ImageNet oracle worst_class over {got['n_rows']} rows", IMAGENET_ORACLE, got["worst_class"])
    _check(f"ImageNet oracle frac_undercovered over {got['n_rows']} rows", IMAGENET_FRAC_UNDER, got["frac_undercovered"])
    return got


def power_curve(n_eval: int, coverage: float, lo: int, hi: int, q: float = SIG_Q) -> dict[str, Any]:
    """Exact power (Binomial tail at the beta-binomial cutoff) at true coverage `coverage` for
    every calibration size lo..hi, summarised: the first size above one half, the size from which
    it stays above one half, the sizes in between where it is not above one half, and the
    maximum."""
    curve = {
        n: float(stats.binom.cdf(significance_cutoff(n_eval, n, ALPHA, q), n_eval, coverage))
        for n in range(lo, hi + 1)
    }
    sizes = sorted(curve)
    first = next(n for n in sizes if curve[n] > 0.5)
    last_not = max((n for n in sizes if curve[n] <= 0.5), default=lo - 1)
    n_max = max(sizes, key=lambda n: curve[n])
    return {
        "n_eval": n_eval,
        "true_coverage": coverage,
        "n_cal_range": [lo, hi],
        "first_above_half": first,
        "power_at_first": curve[first],
        "above_half_from": last_not + 1,
        "not_above_half_after_first": [n for n in range(first, last_not + 1) if curve[n] <= 0.5],
        "max_power": curve[n_max],
        "first_size_at_max": n_max,
        "power_at_n_cal_50": curve.get(50),
    }


def _check(label: str, typed: float, read: float) -> None:
    print(f"{label}: typed {typed:.3f} read {read:.4f}")
    if round(read, 3) != round(typed, 3):
        raise SystemExit(f"{label}: typed value {typed} disagrees with artifact {read:.4f}")


def make_row(
    setting: str,
    K: int,
    n_cal: int | np.ndarray,
    n_eval: int | np.ndarray,
    seeds: int,
    reps: int,
    paper_oracle: float | None,
    paper_source: str,
    paper_n_averaged: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    fl = expected_min_coverage(K, n_cal, n_eval, alpha=ALPHA, reps=reps, seed=0)
    sa_reps = max(reps // SEED_AVG_REPS_DIVISOR, 1000)
    sa = seed_averaged_floor(K, n_cal, n_eval, seeds=seeds, alpha=ALPHA, reps=sa_reps, seed=0)
    per_class = isinstance(n_cal, np.ndarray)
    if per_class:
        fp = [undercovered_false_positive_rate(int(c), int(e), ALPHA, LEVEL) for c, e in zip(n_cal, n_eval)]
        fp_rate = float(np.mean(fp))
        cuts = [significance_cutoff(int(e), int(c), ALPHA, SIG_Q) for c, e in zip(n_cal, n_eval)]
        powers = [float(stats.binom.cdf(k, int(e), POWER_COVERAGE)) for k, e in zip(cuts, n_eval)]
        cut: Any = cuts
        power: Any = powers
    else:
        fp_rate = undercovered_false_positive_rate(int(n_cal), int(n_eval), ALPHA, LEVEL)
        cut = significance_cutoff(int(n_eval), int(n_cal), ALPHA, SIG_Q)
        power = float(stats.binom.cdf(cut, int(n_eval), POWER_COVERAGE))
    z = None if paper_oracle is None else (paper_oracle - fl.mean) / fl.sd
    z_seed = None if paper_oracle is None else (paper_oracle - fl.mean) / (fl.sd / np.sqrt(paper_n_averaged))
    row = {
        "setting": setting,
        "K": K,
        "n_cal": n_cal.tolist() if per_class else int(n_cal),
        "n_eval": n_eval.tolist() if per_class else int(n_eval),
        "floor_mean": fl.mean,
        "floor_sd": fl.sd,
        "floor_q05": fl.q05,
        "floor_q95": fl.q95,
        "seed_avg_floor": sa.mean,
        "seed_avg_seeds": seeds,
        "seed_avg_reps": sa_reps,
        "floor_reps": reps,
        "fp_rate_at_0.90": fp_rate,
        "sig_cutoff": cut,
        "sig_power_at_0.80": power,
        "paper_oracle": paper_oracle,
        "paper_source": paper_source,
        "paper_n_averaged": paper_n_averaged,
        "z": z,
        "z_seed_mean": z_seed,
    }
    row.update(extra)
    return row


def build_rows(reps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = paper_value_counts()
    ks = k_sweep_oracle_means()
    for k, typed in K_SWEEP_TYPED.items():
        _check(f"K-sweep oracle K={k}", typed, ks[k])
    for k in (10, 25, 50, 100):
        if k in K_SWEEP_TYPED:
            oracle, src = K_SWEEP_TYPED[k], K_SWEEP_SOURCE
        else:
            oracle = round(ks[k], 3)
            src = "outputs/analysis/controlled_k_sweep_<cache>.json, oracle mean over five caches"
        rows.append(
            make_row(
                "k_sweep", k, 50, 50, SEED_AVG_SEEDS_KSWEEP, reps, oracle, src,
                paper_n_averaged=counts["k_sweep"][k],
            )
        )

    lab_or = label_sweep_values("oracle")
    lab_ppi = label_sweep_values("ppi")
    for m, typed in LABEL_TYPED.items():
        _check(f"label sweep oracle m={m}", typed, lab_or[m])
        rows.append(
            make_row(
                "label_sweep", 100, m, 50, SEED_AVG_SEEDS, reps, typed,
                f"outputs/analysis/{LABEL_CACHE}, method oracle",
                paper_n_averaged=counts["label_sweep"],
                paper_ppi=round(lab_ppi[m], 3),
            )
        )

    im = check_imagenet()
    rows.append(
        make_row(
            "imagenet1k", 1000, 25, 25, SEED_AVG_SEEDS, reps, IMAGENET_ORACLE,
            IMAGENET_SOURCE + f"; cross-checked against the mean of the {im['n_rows']} leaderboard.md "
            "rows whose dataset starts with imagenet1k_ and whose method is oracle",
            paper_n_averaged=counts["imagenet1k"],
            paper_frac_undercovered=IMAGENET_FRAC_UNDER,
        )
    )

    for name, K, n in (("cifar10c", 10, 500), ("cifar100c", 100, 50)):
        rows.append(make_row(name, K, n, n, SEED_AVG_SEEDS, reps, None, "varies by corruption cell"))
    rows.append(
        make_row("ntu60_xsub", 60, 137, 137, SEED_AVG_SEEDS, reps,
                 leaderboard_oracle("ntu60_xsub"), "leaderboard.md, ntu60_xsub oracle row",
                 paper_n_averaged=counts["leaderboard"]["ntu60_xsub"])
    )
    nucla_cal, nucla_eval = nucla_sizes()
    rows.append(
        make_row("ntu2nucla", 5, nucla_cal, nucla_eval, SEED_AVG_SEEDS, reps,
                 leaderboard_oracle("ntu2nucla"), "leaderboard.md, ntu2nucla oracle row",
                 paper_n_averaged=counts["leaderboard"]["ntu2nucla"])
    )
    return rows


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, list):
        return " / ".join(_f(v, nd) for v in x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_md(rows: list[dict[str, Any]], reps: int, pc: dict[str, Any]) -> str:
    by = {(r["setting"], r["K"], str(r["n_cal"])): r for r in rows}
    k10, k100 = by[("k_sweep", 10, "50")], by[("k_sweep", 100, "50")]
    im = by[("imagenet1k", 1000, "25")]
    sa_reps = sorted({r["seed_avg_reps"] for r in rows})
    def _n(setting: str) -> str:
        vals = sorted({r["paper_n_averaged"] for r in rows if r["setting"] == setting})
        return " and ".join(str(v) for v in vals)

    lb_n = sorted({r["paper_n_averaged"] for r in rows if r["setting"] in LEADERBOARD_ARTIFACT_ROWS})
    n_im_rows = im["paper_n_averaged"] // LEADERBOARD_ROW_SEEDS
    intro = (
        "The validity floor is the expected worst-class coverage that an exactly valid per-class "
        "conformal method still reads at a given number of classes, per-class calibration size and "
        "per-class evaluation size, because the minimum of K noisy per-class coverages sits below the "
        f"target even when every class is covered at level {1 - ALPHA:.2f} in expectation. "
        f"The held-out oracle values in the paper sit on this floor: {k10['paper_oracle']:.3f} against "
        f"a floor of {k10['floor_mean']:.3f} at K equal to 10, {k100['paper_oracle']:.3f} against "
        f"{k100['floor_mean']:.3f} at K equal to 100, and {im['paper_oracle']:.3f} against "
        f"{im['floor_mean']:.3f} on ImageNet at K equal to 1000. "
        f"Under exact validity with {im['n_cal']} calibration and {im['n_eval']} evaluation points per "
        f"class, a fraction {im['fp_rate_at_0.90']:.3f} of the classes read below {LEVEL:.2f} by chance "
        f"alone, so the paper's ImageNet undercovered fraction of {IMAGENET_FRAC_UNDER:.3f} is the chance "
        "level and carries no evidence of undercoverage.\n"
    )
    defs = (
        "The columns are defined as follows. "
        "The floor, with its mean, standard deviation and 5 and 95 percent quantiles, is the expected "
        "minimum over the K classes of the empirical per-class coverages of a single calibration and "
        f"evaluation draw, estimated from {reps} Monte Carlo repetitions with seed 0 at alpha equal to "
        f"{ALPHA}. "
        "The column seed_avg_floor first averages each class's empirical coverage over the stated number "
        "of independent calibration and evaluation draws and only then takes the minimum over classes, "
        f"estimated from {' and '.join(str(v) for v in sa_reps)} Monte Carlo repetitions with seed 0. "
        "That is a different estimand from the paper's, which takes the minimum within each draw first "
        "and then averages those worst-class values over seeds, so it is reported as a candidate "
        "estimand for a future release and is never compared with the paper's oracle values. "
        f"The column fp_rate_at_0.90 is the exact probability that a class covered validly reads below "
        f"{LEVEL:.2f}, computed from the beta-binomial law with no simulation. "
        f"The column sig_cutoff is the largest covered count whose probability is at most {SIG_Q} under "
        "the beta-binomial law of a valid per-class calibration on n_cal points evaluated on n_eval "
        "points, so a class at or below it is undercovered beyond what calibration and evaluation noise "
        f"explain. The column sig_power_at_0.80 is the probability that a class whose true coverage is "
        f"{POWER_COVERAGE:.2f} falls at or below that cutoff. "
        "The column z is the paper oracle minus the floor mean, divided by the single-draw floor standard "
        "deviation. The column z_seed_mean divides the same difference by the floor standard deviation "
        "over the square root of the number of worst-class values the paper value averages, counted from "
        f"the committed artifacts. That number is {_n('k_sweep')} for the K sweep rows, which average "
        f"{len(K_SWEEP_CACHES)} caches by {k10['paper_n_averaged'] // len(K_SWEEP_CACHES)} "
        f"subset seeds, {_n('label_sweep')} for the label sweep rows, {_n('imagenet1k')} for the ImageNet "
        f"row, which averages {n_im_rows} leaderboard rows of {LEADERBOARD_ROW_SEEDS} seeds each, and "
        f"{' and '.join(str(v) for v in lb_n)} for the leaderboard rows. "
        "Exact validity gives each averaged value the floor law. Independence between the averaged values "
        f"is an assumption. The K sweep caches share the class subset and the target split at each subset "
        f"seed, so {_n('k_sweep')} may overstate the independent count for those rows, and the ImageNet rows "
        "are corruption cells of one image set under each backbone and may share splits as well. Positive "
        "dependence shrinks the effective count and moves every z_seed_mean toward zero, so the reading that "
        "the oracle sits on the floor stands under either count. The paper's oracle is a mean over those "
        "values, so z_seed_mean is the one that compares like with like. "
        "For ntu2nucla the per-class sizes are listed, the evaluation count of each class is half its "
        "target count rounded as the runner rounds it and the calibration count is the remaining labeled "
        "pool, the false-positive rate is the mean over its classes, and the cutoff and power are given "
        "per class. The label sweep rows carry the PPI value from the same artifact in the JSON file as "
        "paper_ppi.\n"
    )
    dips = pc["not_above_half_after_first"]
    power = (
        f"\nPower of the significance test at true coverage {pc['true_coverage']:.2f} with "
        f"{pc['n_eval']} evaluation points, over calibration sizes {pc['n_cal_range'][0]} to "
        f"{pc['n_cal_range'][1]}, was computed exactly at every size. It first exceeds one half at "
        f"{pc['first_above_half']} calibration points, where it is {pc['power_at_first']:.3f}. It is not "
        f"monotone, because it falls back to one half or below at {len(dips)} larger sizes up to "
        f"{max(dips) if dips else pc['first_above_half']}. It stays above one half from "
        f"{pc['above_half_from']} calibration points onward and never exceeds {pc['max_power']:.3f}. "
        f"At {pc['n_cal_range'][0]} calibration points it is {pc['power_at_n_cal_50']:.3f}.\n"
    )
    head = (
        "| setting | K | n_cal | n_eval | floor_mean | floor_sd | floor_q05 | floor_q95 | "
        "seed_avg_floor | fp_rate_at_0.90 | sig_cutoff | sig_power_at_0.80 | paper_oracle | z | "
        "z_seed_mean |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    body = "".join(
        f"| {r['setting']} | {r['K']} | {_f(r['n_cal'])} | {_f(r['n_eval'])} | {_f(r['floor_mean'])} | "
        f"{_f(r['floor_sd'])} | {_f(r['floor_q05'])} | {_f(r['floor_q95'])} | "
        f"{_f(r['seed_avg_floor'])} ({r['seed_avg_seeds']} seeds) | {_f(r['fp_rate_at_0.90'])} | "
        f"{_f(r['sig_cutoff'])} | {_f(r['sig_power_at_0.80'])} | {_f(r['paper_oracle'])} | "
        f"{_f(r['z'], 2)} | {_f(r['z_seed_mean'], 2)} |\n"
        for r in rows
    )
    return "# Validity floor of worst-class coverage\n\n" + intro + "\n" + defs + "\n" + head + body + power


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args(argv)
    rows = build_rows(args.reps)
    pc = power_curve(POWER_N_EVAL, POWER_COVERAGE, *POWER_CAL_RANGE)
    out, md = Path(args.out), Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    payload = {"alpha": ALPHA, "level": LEVEL, "reps": args.reps, "seed": 0, "rows": rows, "power_curve": pc}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    md.write_text(render_md(rows, args.reps, pc))
    for r in rows:
        print(
            f"{r['setting']:>11} K={r['K']:<4} n_cal={_f(r['n_cal']):<22} floor={r['floor_mean']:.4f} "
            f"sd={r['floor_sd']:.4f} seedavg={r['seed_avg_floor']:.4f} fp={r['fp_rate_at_0.90']:.4f} "
            f"cut={_f(r['sig_cutoff'])} power={_f(r['sig_power_at_0.80'])} oracle={_f(r['paper_oracle'])} z={_f(r['z'], 2)} "
            f"z_seed_mean={_f(r['z_seed_mean'], 2)}"
            + (f" ppi={r['paper_ppi']}" if "paper_ppi" in r else "")
        )
    print(
        f"power curve: first above half {pc['first_above_half']} ({pc['power_at_first']:.4f}), "
        f"above half from {pc['above_half_from']}, max {pc['max_power']:.4f}, "
        f"not above half after first at {pc['not_above_half_after_first']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
