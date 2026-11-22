#!/usr/bin/env python3
"""Bootstrap intervals and floor excess for the controlled K-sweep and the label-complexity sweep."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from pcpbench.metrics.validity_floor import expected_min_coverage


MODERATE_CACHES: tuple[str, ...] = (
    "contrast_s1", "contrast_s2", "contrast_s3", "brightness_s1", "brightness_s3",
)
FLOORED_CACHES: tuple[str, ...] = ("gaussian_noise_s3", "fog_s3", "jpeg_compression_s1")
CONTROLLED_METHODS: tuple[str, ...] = ("split", "mondrian", "weighted", "ppi", "acp", "oracle")
NON_ORACLE_METHODS: tuple[str, ...] = ("split", "mondrian", "weighted", "ppi", "acp")
PAPER_RANGE_END_METHODS: tuple[str, str] = ("mondrian", "ppi")
K_VALUES: tuple[int, ...] = (10, 25, 50, 100)
N_SUBSET_SEEDS = 5
N_CAL_PER_CLASS = 50
N_EVAL_PER_CLASS = 50
ALPHA = 0.10

LABEL_SWEEP_CACHE = "contrast_s3"
LABEL_SWEEP_METHODS: tuple[str, ...] = ("ppi", "oracle", "acp")
K_LABEL_SWEEP = 100
N_EVAL_LABEL_SWEEP = 50

DEFAULT_B = 10000
DEFAULT_SEED = 0
CI_LO_PCT, CI_HI_PCT = 2.5, 97.5


@dataclass(frozen=True)
class PerKStat:
    method: str
    k: int
    n_records: int
    mean: float
    se: float
    ci_lo: float
    ci_hi: float
    floor_mean: float
    floor_sd: float
    excess: float


@dataclass(frozen=True)
class DecayStat:
    method: str
    decay_mean: float
    decay_se: float
    decay_ci_lo: float
    decay_ci_hi: float
    ratio_to_oracle: float
    ratio_se: float
    ratio_ci_lo: float
    ratio_ci_hi: float
    ratio_flag_at_or_below_one: bool
    paper_convention_ratio: float


@dataclass(frozen=True)
class RatioSeCaveat:
    """The near-zero and negative oracle-decay draw counts, counted inside the same five paired
    bootstrap draw sets that produce each method's own ratio interval and standard error (never a
    separate draw of the oracle's decay on its own), one count per method in the order given by
    ``methods``."""

    B: int
    methods: tuple[str, ...]
    n_oracle_decay_draws_le_0_01: tuple[int, ...]
    n_oracle_decay_draws_le_0: tuple[int, ...]


@dataclass(frozen=True)
class VsOracleStat:
    method: str
    k: int
    n_records: int
    diff_mean: float
    diff_se: float
    diff_ci_lo: float
    diff_ci_hi: float
    n_below_oracle: int


@dataclass(frozen=True)
class FloorCheck:
    method: str
    k: int
    below_floor_mean: bool


@dataclass(frozen=True)
class FlooredCacheStat:
    cache: str
    k: int
    n_seeds: int
    mean: float
    se: float
    ci_lo: float
    ci_hi: float
    floor_mean: float
    floor_sd: float


@dataclass(frozen=True)
class LabelSweepStat:
    method: str
    m: int
    n_seeds: int
    mean: float | None
    se: float | None
    ci_lo: float | None
    ci_hi: float | None
    floor_mean: float | None
    floor_sd: float | None
    excess: float | None
    labels_per_class_drawn: int | None
    cal_source: str | None
    reused_from_m: int | None


@dataclass(frozen=True)
class LabelPairStat:
    m: int
    n_seeds: int
    diff_mean: float
    diff_se: float
    diff_ci_lo: float
    diff_ci_hi: float
    reused_from_m: int | None


def load_controlled_records(analysis_dir: Path, cache: str) -> list[dict]:
    path = analysis_dir / f"controlled_k_sweep_{cache}.json"
    return json.loads(path.read_text())["records"]


def load_label_sweep(analysis_dir: Path, cache: str) -> dict:
    path = analysis_dir / f"label_complexity_{cache}.json"
    return json.loads(path.read_text())


def index_controlled_records(
    records_by_cache: dict[str, list[dict]],
) -> dict[str, dict[int, dict[str, dict[int, float]]]]:
    """method -> k -> cache -> seed -> worst_class, one record per cell."""
    idx: dict = {}
    for cache, records in records_by_cache.items():
        for r in records:
            method, k, seed = r["method"], r["k"], r["subset_seed"]
            slot = idx.setdefault(method, {}).setdefault(k, {}).setdefault(cache, {})
            if seed in slot:
                raise ValueError(f"duplicate record for method={method} k={k} cache={cache} seed={seed}")
            slot[seed] = r["worst_class"]
    return idx


def matrix_for(
    idx: dict, method: str, k: int, caches: Sequence[str], n_seeds: int
) -> np.ndarray:
    """(n_caches, n_seeds) grid of worst_class values, seed index ascending."""
    out = np.empty((len(caches), n_seeds), dtype=float)
    for ci, cache in enumerate(caches):
        cell = idx[method][k][cache]
        for si in range(n_seeds):
            out[ci, si] = cell[si]
    return out


def bootstrap_two_level_indices(
    n_cache: int, n_seed: int, rng: np.random.Generator, B: int
) -> tuple[np.ndarray, np.ndarray]:
    cache_idx = rng.integers(0, n_cache, size=(B, n_cache))
    seed_idx = rng.integers(0, n_seed, size=(B, n_cache, n_seed))
    return cache_idx, seed_idx


def apply_two_level(matrix: np.ndarray, cache_idx: np.ndarray, seed_idx: np.ndarray) -> np.ndarray:
    vals = matrix[cache_idx[:, :, None], seed_idx]
    return vals.mean(axis=(1, 2))


def bootstrap_two_level_mean(matrix: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    n_cache, n_seed = matrix.shape
    cache_idx, seed_idx = bootstrap_two_level_indices(n_cache, n_seed, rng, B)
    return apply_two_level(matrix, cache_idx, seed_idx)


def bootstrap_two_level_paired(
    matrix_a: np.ndarray, matrix_b: np.ndarray, rng: np.random.Generator, B: int
) -> tuple[np.ndarray, np.ndarray]:
    """Both matrices resampled with the SAME drawn cache and seed indices."""
    n_cache, n_seed = matrix_a.shape
    cache_idx, seed_idx = bootstrap_two_level_indices(n_cache, n_seed, rng, B)
    return apply_two_level(matrix_a, cache_idx, seed_idx), apply_two_level(matrix_b, cache_idx, seed_idx)


def bootstrap_one_level_indices(n: int, rng: np.random.Generator, B: int) -> np.ndarray:
    return rng.integers(0, n, size=(B, n))


def apply_one_level(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return values[idx].mean(axis=1)


def bootstrap_one_level_mean(values: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    idx = bootstrap_one_level_indices(values.shape[0], rng, B)
    return apply_one_level(values, idx)


def bootstrap_one_level_paired(
    values_a: np.ndarray, values_b: np.ndarray, rng: np.random.Generator, B: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = bootstrap_one_level_indices(values_a.shape[0], rng, B)
    return apply_one_level(values_a, idx), apply_one_level(values_b, idx)


def count_draws_at_or_below(draws: np.ndarray, threshold: float) -> int:
    """Count of bootstrap draws at or below ``threshold``, a pure counting primitive behind the
    ratio-SE caveat (the near-zero and negative oracle-decay draw counts)."""
    return int(np.sum(draws <= threshold))


def summarize(point: float, draws: np.ndarray) -> dict:
    return {
        "mean": float(point),
        "se": float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0,
        "ci_lo": float(np.percentile(draws, CI_LO_PCT)),
        "ci_hi": float(np.percentile(draws, CI_HI_PCT)),
    }


def controlled_per_k(
    idx: dict,
    caches: Sequence[str],
    methods: Sequence[str],
    k_values: Sequence[int],
    n_seeds: int,
    rng: np.random.Generator,
    B: int,
) -> list[PerKStat]:
    out: list[PerKStat] = []
    for method in methods:
        for k in k_values:
            mat = matrix_for(idx, method, k, caches, n_seeds)
            point = float(mat.mean())
            draws = bootstrap_two_level_mean(mat, rng, B)
            s = summarize(point, draws)
            floor = expected_min_coverage(k, N_CAL_PER_CLASS, N_EVAL_PER_CLASS, alpha=ALPHA)
            out.append(PerKStat(
                method=method, k=k, n_records=int(mat.size),
                mean=s["mean"], se=s["se"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                floor_mean=floor.mean, floor_sd=floor.sd, excess=point - floor.mean,
            ))
    return out


def controlled_decay(
    idx: dict,
    caches: Sequence[str],
    methods: Sequence[str],
    n_seeds: int,
    rng: np.random.Generator,
    B: int,
) -> tuple[list[DecayStat], RatioSeCaveat]:
    """Two decay-ratio conventions per method: the unrounded ratio of the method's and the
    oracle's full-precision 25-record decay means (with its own bootstrap interval), and the
    displayed-endpoint convention, which rounds each method's K=10 and K=100 mean to three
    decimals before subtracting and dividing."""
    mat10 = {method: matrix_for(idx, method, 10, caches, n_seeds) for method in methods}
    mat100 = {method: matrix_for(idx, method, 100, caches, n_seeds) for method in methods}
    decay_matrices = {method: mat10[method] - mat100[method] for method in methods}
    oracle_decay_point = float(decay_matrices["oracle"].mean())
    oracle_paper_decay = round(float(mat10["oracle"].mean()), 3) - round(float(mat100["oracle"].mean()), 3)

    out: list[DecayStat] = []
    caveat_methods: list[str] = []
    caveat_le_0_01: list[int] = []
    caveat_le_0: list[int] = []
    for method in methods:
        if method == "oracle":
            continue
        dm = decay_matrices[method]
        decay_point = float(dm.mean())
        decay_draws = bootstrap_two_level_mean(dm, rng, B)
        decay_s = summarize(decay_point, decay_draws)

        ratio_point = decay_point / oracle_decay_point
        method_draws, oracle_draws = bootstrap_two_level_paired(dm, decay_matrices["oracle"], rng, B)
        ratio_draws = method_draws / oracle_draws
        ratio_se = float(np.std(ratio_draws, ddof=1))
        ratio_lo = float(np.percentile(ratio_draws, CI_LO_PCT))
        ratio_hi = float(np.percentile(ratio_draws, CI_HI_PCT))

        method_paper_decay = round(float(mat10[method].mean()), 3) - round(float(mat100[method].mean()), 3)
        paper_convention_ratio = method_paper_decay / oracle_paper_decay

        out.append(DecayStat(
            method=method,
            decay_mean=decay_s["mean"], decay_se=decay_s["se"],
            decay_ci_lo=decay_s["ci_lo"], decay_ci_hi=decay_s["ci_hi"],
            ratio_to_oracle=ratio_point, ratio_se=ratio_se,
            ratio_ci_lo=ratio_lo, ratio_ci_hi=ratio_hi,
            ratio_flag_at_or_below_one=ratio_lo <= 1.0,
            paper_convention_ratio=paper_convention_ratio,
        ))
        caveat_methods.append(method)
        caveat_le_0_01.append(count_draws_at_or_below(oracle_draws, 0.01))
        caveat_le_0.append(count_draws_at_or_below(oracle_draws, 0.0))

    caveat = RatioSeCaveat(
        B=B,
        methods=tuple(caveat_methods),
        n_oracle_decay_draws_le_0_01=tuple(caveat_le_0_01),
        n_oracle_decay_draws_le_0=tuple(caveat_le_0),
    )
    return out, caveat


def controlled_vs_oracle(
    idx: dict,
    caches: Sequence[str],
    methods: Sequence[str],
    k_values: Sequence[int],
    n_seeds: int,
    rng: np.random.Generator,
    B: int,
) -> list[VsOracleStat]:
    out: list[VsOracleStat] = []
    for method in methods:
        if method == "oracle":
            continue
        for k in k_values:
            mat_m = matrix_for(idx, method, k, caches, n_seeds)
            mat_o = matrix_for(idx, "oracle", k, caches, n_seeds)
            diff = mat_m - mat_o
            diff_point = float(diff.mean())
            draws = bootstrap_two_level_mean(diff, rng, B)
            s = summarize(diff_point, draws)
            out.append(VsOracleStat(
                method=method, k=k, n_records=int(diff.size),
                diff_mean=s["mean"], diff_se=s["se"], diff_ci_lo=s["ci_lo"], diff_ci_hi=s["ci_hi"],
                n_below_oracle=int(np.sum(diff < 0)),
            ))
    return out


def controlled_floor_check(per_k_stats: Sequence[PerKStat], k: int = 100) -> list[FloorCheck]:
    out: list[FloorCheck] = []
    for s in per_k_stats:
        if s.k == k and s.method != "oracle":
            out.append(FloorCheck(method=s.method, k=k, below_floor_mean=s.ci_hi < s.floor_mean))
    return out


def oracle_values(records: Sequence[dict], k: int) -> np.ndarray:
    by_seed = {r["subset_seed"]: r["worst_class"] for r in records if r["method"] == "oracle" and r["k"] == k}
    return np.array([by_seed[s] for s in sorted(by_seed)], dtype=float)


def floored_cache_stats(
    records_by_cache: dict[str, list[dict]],
    caches: Sequence[str],
    rng: np.random.Generator,
    B: int,
) -> list[FlooredCacheStat]:
    out: list[FlooredCacheStat] = []
    for cache in caches:
        records = records_by_cache[cache]
        for k in (10, 100):
            vals = oracle_values(records, k)
            point = float(vals.mean())
            draws = bootstrap_one_level_mean(vals, rng, B)
            s = summarize(point, draws)
            floor = expected_min_coverage(k, N_CAL_PER_CLASS, N_EVAL_PER_CLASS, alpha=ALPHA)
            out.append(FlooredCacheStat(
                cache=cache, k=k, n_seeds=int(vals.shape[0]),
                mean=s["mean"], se=s["se"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                floor_mean=floor.mean, floor_sd=floor.sd,
            ))
    return out


def index_label_records(per_seed: Sequence[dict]) -> dict[str, dict[int, dict[int, float | None]]]:
    idx: dict = {}
    for r in per_seed:
        idx.setdefault(r["method"], {}).setdefault(r["m"], {})[r["seed"]] = r["worst_class"]
    return idx


def index_label_cal_counts(per_seed: Sequence[dict]) -> dict[str, dict[int, dict[int, list[int]]]]:
    """method -> m -> seed -> the run's actual per_class_labeled array (length K)."""
    idx: dict = {}
    for r in per_seed:
        idx.setdefault(r["method"], {}).setdefault(r["m"], {})[r["seed"]] = r["per_class_labeled"]
    return idx


def _label_drawn_count(arr: np.ndarray) -> int:
    vals = np.unique(arr)
    if vals.size != 1:
        raise ValueError(
            "per_class_labeled is not constant across classes; the 'labels per class "
            "drawn' column assumes a single count per method and m"
        )
    return int(vals[0])


def _canonical_cal_array(
    cal_idx: dict, methods: Sequence[str], m: int
) -> tuple[np.ndarray | None, bool]:
    arrays = [
        np.asarray(arr, dtype=np.int64)
        for method in methods
        for arr in cal_idx[method][m].values()
    ]
    first = arrays[0]
    agree = all(np.array_equal(first, a) for a in arrays[1:])
    return (first, True) if agree else (None, False)


def _per_seed_fallback_floor(
    cal_idx: dict, method: str, m: int, K: int, n_eval: int, alpha: float
) -> tuple[float, float]:
    seed_map = cal_idx[method][m]
    means, sds = [], []
    for seed in sorted(seed_map):
        arr = np.asarray(seed_map[seed], dtype=np.int64)
        floor = expected_min_coverage(K, arr, n_eval, alpha=alpha)
        means.append(floor.mean)
        sds.append(floor.sd)
    return float(np.mean(means)), float(np.mean(sds))


def _method_own_drawn_count(cal_idx: dict, method: str, m: int) -> int | None:
    arrays = [np.asarray(arr, dtype=np.int64) for arr in cal_idx[method][m].values()]
    if not all(np.array_equal(arrays[0], a) for a in arrays[1:]):
        return None
    try:
        return _label_drawn_count(arrays[0])
    except ValueError:
        return None


def label_sweep_stats(
    payload: dict,
    methods: Sequence[str],
    rng: np.random.Generator,
    B: int,
) -> tuple[list[LabelSweepStat], list[LabelPairStat], int | None, int | None]:
    m_values: list[int] = payload["m_values"]
    idx = index_label_records(payload["per_seed"])
    cal_idx = index_label_cal_counts(payload["per_seed"])

    stats: list[LabelSweepStat] = []
    history: dict[str, list[tuple[int, tuple, tuple, LabelSweepStat]]] = {meth: [] for meth in methods}
    for method in methods:
        for m in m_values:
            seed_map = idx[method][m]
            seeds_sorted = sorted(seed_map)
            raw = [seed_map[s] for s in seeds_sorted]
            if any(v is None for v in raw):
                stat = LabelSweepStat(
                    method=method, m=m, n_seeds=0,
                    mean=None, se=None, ci_lo=None, ci_hi=None,
                    floor_mean=None, floor_sd=None, excess=None,
                    labels_per_class_drawn=None, cal_source=None, reused_from_m=None,
                )
                stats.append(stat)
                continue

            cal_seed_map = cal_idx[method][m]
            raw_tuple = tuple(raw)
            cal_tuple = tuple(tuple(cal_seed_map[s]) for s in seeds_sorted)
            match = next(
                (entry for entry in history[method] if entry[1] == raw_tuple and entry[2] == cal_tuple),
                None,
            )
            if match is not None:
                source_m, _, _, source_stat = match
                stat = replace(source_stat, m=m, reused_from_m=source_m)
            else:
                vals = np.array(raw, dtype=float)
                point = float(vals.mean())
                draws = bootstrap_one_level_mean(vals, rng, B)
                s = summarize(point, draws)
                if m >= 1:
                    canonical, agree = _canonical_cal_array(cal_idx, methods, m)
                    if agree:
                        cal_source = "agreed_across_seeds_and_methods"
                        floor = expected_min_coverage(K_LABEL_SWEEP, canonical, N_EVAL_LABEL_SWEEP, alpha=ALPHA)
                        floor_mean, floor_sd = floor.mean, floor.sd
                        drawn = _label_drawn_count(canonical)
                    else:
                        cal_source = "fallback_per_seed_average"
                        floor_mean, floor_sd = _per_seed_fallback_floor(
                            cal_idx, method, m, K_LABEL_SWEEP, N_EVAL_LABEL_SWEEP, ALPHA
                        )
                        drawn = _method_own_drawn_count(cal_idx, method, m)
                    excess = point - floor_mean
                else:
                    floor_mean = floor_sd = excess = None
                    canonical, agree = _canonical_cal_array(cal_idx, methods, m)
                    if agree:
                        cal_source = "agreed_across_seeds_and_methods"
                        drawn = _label_drawn_count(canonical)
                    else:
                        cal_source = "fallback_per_seed_average"
                        drawn = _method_own_drawn_count(cal_idx, method, m)
                stat = LabelSweepStat(
                    method=method, m=m, n_seeds=int(vals.shape[0]),
                    mean=s["mean"], se=s["se"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                    floor_mean=floor_mean, floor_sd=floor_sd, excess=excess,
                    labels_per_class_drawn=drawn, cal_source=cal_source, reused_from_m=None,
                )
            stats.append(stat)
            history[method].append((m, raw_tuple, cal_tuple, stat))

    pair_stats: list[LabelPairStat] = []
    pair_history: list[tuple[int, tuple, tuple, LabelPairStat]] = []
    for m in m_values:
        ppi_map, oracle_map = idx["ppi"][m], idx["oracle"][m]
        seeds_sorted = sorted(set(ppi_map) & set(oracle_map))
        ppi_raw = [ppi_map[s] for s in seeds_sorted]
        oracle_raw = [oracle_map[s] for s in seeds_sorted]
        if any(v is None for v in ppi_raw) or any(v is None for v in oracle_raw):
            continue
        ppi_tuple, oracle_tuple = tuple(ppi_raw), tuple(oracle_raw)
        match = next(
            (entry for entry in pair_history if entry[1] == ppi_tuple and entry[2] == oracle_tuple),
            None,
        )
        if match is not None:
            source_m, _, _, source_pair = match
            pair = replace(source_pair, m=m, reused_from_m=source_m)
        else:
            diff = np.array(ppi_raw, dtype=float) - np.array(oracle_raw, dtype=float)
            diff_point = float(diff.mean())
            draws = bootstrap_one_level_mean(diff, rng, B)
            s = summarize(diff_point, draws)
            pair = LabelPairStat(
                m=m, n_seeds=int(diff.shape[0]),
                diff_mean=s["mean"], diff_se=s["se"], diff_ci_lo=s["ci_lo"], diff_ci_hi=s["ci_hi"],
                reused_from_m=None,
            )
        pair_stats.append(pair)
        pair_history.append((m, ppi_tuple, oracle_tuple, pair))

    def smallest_m_meets_floor(method: str) -> int | None:
        for m in sorted(x for x in m_values if x >= 1):
            entry = next(st for st in stats if st.method == method and st.m == m)
            if entry.floor_mean is None or entry.ci_lo is None or entry.ci_hi is None:
                continue
            if entry.ci_lo <= entry.floor_mean <= entry.ci_hi:
                return m
        return None

    smallest_ppi = smallest_m_meets_floor("ppi")
    smallest_oracle = smallest_m_meets_floor("oracle")
    return stats, pair_stats, smallest_ppi, smallest_oracle


def to_json_payload(
    *,
    B: int,
    seed: int,
    moderate_caches: Sequence[str],
    floored_caches: Sequence[str],
    label_sweep_cache: str,
    per_k: Sequence[PerKStat],
    decay: Sequence[DecayStat],
    vs_oracle: Sequence[VsOracleStat],
    floor_check: Sequence[FloorCheck],
    floored: Sequence[FlooredCacheStat],
    label_stats: Sequence[LabelSweepStat],
    label_pairs: Sequence[LabelPairStat],
    smallest_ppi: int | None,
    smallest_oracle: int | None,
    ratio_se_caveat: RatioSeCaveat,
) -> dict:
    return {
        "config": {
            "B": B, "seed": seed, "alpha": ALPHA,
            "n_cal_per_class": N_CAL_PER_CLASS, "n_eval_per_class": N_EVAL_PER_CLASS,
            "moderate_caches": list(moderate_caches), "floored_caches": list(floored_caches),
            "label_sweep_cache": label_sweep_cache, "k_label_sweep": K_LABEL_SWEEP,
            "n_eval_label_sweep": N_EVAL_LABEL_SWEEP,
        },
        "controlled": {
            "per_k": [asdict(s) for s in per_k],
            "decay": [asdict(s) for s in decay],
            "vs_oracle": [asdict(s) for s in vs_oracle],
            "floor_check_k100": [asdict(s) for s in floor_check],
            "ratio_se_caveat": asdict(ratio_se_caveat),
        },
        "floored_caches": [asdict(s) for s in floored],
        "label_sweep": {
            "per_m": [asdict(s) for s in label_stats],
            "ppi_vs_oracle": [asdict(s) for s in label_pairs],
            "smallest_m_ppi_meets_floor": smallest_ppi,
            "smallest_m_oracle_meets_floor": smallest_oracle,
        },
    }


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"{_fmt(lo)} to {_fmt(hi)}"


def _join_and(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _range_end_sentences(decay: Sequence[DecayStat]) -> tuple[str, set[str]]:
    by_method = {d.method: d for d in decay}
    missing = [method for method in PAPER_RANGE_END_METHODS if method not in by_method]
    if missing:
        raise ValueError(f"no decay ratio for {_join_and(missing)}, whose ratios the paper printed")
    lo_end, hi_end = (by_method[method] for method in PAPER_RANGE_END_METHODS)
    lo, hi = lo_end.paper_convention_ratio, hi_end.paper_convention_ratio
    if lo > hi:
        raise ValueError(f"the ratio of {lo_end.method} ({lo:.3f}) is above that of {hi_end.method} "
                         f"({hi:.3f}), so they are not the lower and the upper end of a range")
    outside = [d for d in decay if not lo <= d.paper_convention_ratio <= hi]
    if not outside:
        return (
            f"The paper prints {lo:.1f} and {hi:.1f} as the ends of its range, the ratios of "
            f"the displayed three-decimal endpoints for {lo_end.method} and {hi_end.method}."
        ), {lo_end.method, hi_end.method}
    outside_part = _join_and([f"{d.paper_convention_ratio:.1f} for {d.method}" for d in outside])
    return (
        "The reviewed version of the paper printed the ratios of the displayed three-decimal "
        f"endpoints for {lo_end.method} and {hi_end.method} as the ends of its range; on these "
        f"records they are {lo:.1f} and {hi:.1f}. On these records the same convention gives "
        f"{outside_part}, outside that range."
    ), {lo_end.method, hi_end.method, *(d.method for d in outside)}


def render_markdown(
    *,
    B: int,
    per_k: Sequence[PerKStat],
    decay: Sequence[DecayStat],
    vs_oracle: Sequence[VsOracleStat],
    floor_check: Sequence[FloorCheck],
    floored: Sequence[FlooredCacheStat],
    label_stats: Sequence[LabelSweepStat],
    label_pairs: Sequence[LabelPairStat],
    smallest_ppi: int | None,
    smallest_oracle: int | None,
    moderate_caches: Sequence[str],
    floored_caches: Sequence[str],
    ratio_se_caveat: RatioSeCaveat,
) -> str:
    lines: list[str] = []
    lines.append("# Bootstrap intervals and floor excess for the controlled sweep and the label sweep")
    lines.append("")
    lines.append(
        "Every mean below is a plain average over the stated records. Every interval is a "
        f"95 percent percentile bootstrap of {B} resamples drawn from a random number "
        "generator seeded at zero. Every interval other than the five decay-ratio "
        "intervals in Table 2 is reported beside its bootstrap standard error, the "
        "standard deviation of the same resampled distribution. The decay-ratio "
        "intervals carry the caveat given after Table 2 in place of a standard error. "
        "The five moderate caches "
        f"are {', '.join(moderate_caches)}, contributing 5 subset seeds each for 25 records "
        "per method and K. The controlled sweep draws a fresh random class subset at every "
        "K, so the sampled classes differ between the K equal to 10 and the K equal to 100 "
        "records of a given subset seed. The pairing used throughout this document is by "
        "seed index alone. A two level bootstrap resamples the five caches with replacement "
        "and then resamples each drawn cache's five seeds with replacement, which is the "
        "procedure behind every controlled sweep interval below. The three floored caches "
        f"are {', '.join(floored_caches)}, each contributing one cache of 5 seeds, so their "
        "intervals use a one level bootstrap over seeds alone. The label sweep intervals "
        "also use a one level bootstrap, over its 10 seeds."
    )
    lines.append("")

    lines.append("## Table 1. Worst-class coverage by method and K, with the validity floor")
    lines.append("")
    lines.append("| method | K | n | mean | SE | 95 percent interval | floor mean | floor SD | excess over floor |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in per_k:
        lines.append(
            f"| {s.method} | {s.k} | {s.n_records} | {_fmt(s.mean)} | {_fmt(s.se)} | "
            f"{_fmt_ci(s.ci_lo, s.ci_hi)} | {_fmt(s.floor_mean)} | {_fmt(s.floor_sd)} | {_fmt(s.excess)} |"
        )
    lines.append("")
    lines.append("### Floored caches, oracle only")
    lines.append("")
    lines.append("| cache | K | n | mean | SE | 95 percent interval | floor mean | floor SD |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in floored:
        lines.append(
            f"| {s.cache} | {s.k} | {s.n_seeds} | {_fmt(s.mean)} | {_fmt(s.se)} | "
            f"{_fmt_ci(s.ci_lo, s.ci_hi)} | {_fmt(s.floor_mean)} | {_fmt(s.floor_sd)} |"
        )
    lines.append("")

    lines.append("## Table 2. Decay from K equal to 10 to K equal to 100, ratio to the oracle decay, and the gap to the oracle at K equal to 100")
    lines.append("")
    lines.append(
        "| method | decay | decay SE | decay 95 percent interval | ratio to oracle decay | "
        "displayed-endpoint convention | ratio 95 percent interval | ratio interval at or below one | "
        "diff to oracle at K=100 | diff 95 percent interval | records below oracle at K=100 of 25 | "
        "K=100 interval below floor mean |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    vs_oracle_k100 = {v.method: v for v in vs_oracle if v.k == 100}
    floor_check_by_method = {f.method: f for f in floor_check}
    for d in decay:
        v = vs_oracle_k100[d.method]
        fc = floor_check_by_method[d.method]
        lines.append(
            f"| {d.method} | {_fmt(d.decay_mean)} | {_fmt(d.decay_se)} | "
            f"{_fmt_ci(d.decay_ci_lo, d.decay_ci_hi)} | {d.ratio_to_oracle:.1f} | {d.paper_convention_ratio:.1f} | "
            f"{d.ratio_ci_lo:.1f} to {d.ratio_ci_hi:.1f} | {str(d.ratio_flag_at_or_below_one).lower()} | "
            f"{_fmt(v.diff_mean)} | {_fmt_ci(v.diff_ci_lo, v.diff_ci_hi)} | {v.n_below_oracle} | "
            f"{str(fc.below_floor_mean).lower()} |"
        )
    lines.append("")

    differ = [d for d in decay if round(d.paper_convention_ratio, 1) != round(d.ratio_to_oracle, 1)]
    agree = [d for d in decay if round(d.paper_convention_ratio, 1) == round(d.ratio_to_oracle, 1)]
    convention_sentence, named = _range_end_sentences(decay)
    inner_differ = [d for d in differ if d.method not in named]
    if inner_differ:
        inner_part = _join_and([f"{d.paper_convention_ratio:.1f} for {d.method}" for d in inner_differ])
        convention_sentence += f" The same displayed-endpoint convention gives {inner_part}."
    if differ:
        convention_sentence += (
            f" The two conventions differ for {_join_and([d.method for d in differ])}, where the "
            f"unrounded ratio of means is {_join_and([f'{d.ratio_to_oracle:.1f}' for d in differ])}"
        )
        if agree:
            convention_sentence += f", and they agree for {_join_and([d.method for d in agree])}."
        else:
            convention_sentence += "."
    else:
        convention_sentence += (
            " The unrounded ratio of means agrees with the displayed-endpoint "
            "ratio, rounded to one decimal, for every method."
        )
    lines.append(convention_sentence)
    lines.append("")
    lines.append(
        "The 95 percent percentile interval is the reported uncertainty for each ratio "
        "because the bootstrap standard error of a ratio is inflated by the draws in "
        "which the oracle's decay falls near zero. In the draws behind the five ratio "
        "intervals the oracle's decay fell at or below 0.01 in "
        f"{_join_and([str(c) for c in ratio_se_caveat.n_oracle_decay_draws_le_0_01])} of "
        f"{ratio_se_caveat.B} draws for {_join_and(list(ratio_se_caveat.methods))}, and "
        "at or below zero in "
        f"{_join_and([str(c) for c in ratio_se_caveat.n_oracle_decay_draws_le_0])}."
    )
    lines.append("")

    lines.append("## Table 3. Label-complexity sweep, worst-class coverage by method and target labels per class")
    lines.append("")
    lines.append(
        "| method | m | labels per class drawn | n | mean | SE | 95 percent interval | "
        "floor mean | floor SD | excess over floor |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in label_stats:
        drawn_s = "undefined" if s.labels_per_class_drawn is None else str(s.labels_per_class_drawn)
        mean_s = "undefined" if s.mean is None else _fmt(s.mean)
        se_s = "undefined" if s.se is None else _fmt(s.se)
        ci_s = "undefined" if s.ci_lo is None else _fmt_ci(s.ci_lo, s.ci_hi)
        floor_mean_s = "undefined" if s.floor_mean is None else _fmt(s.floor_mean)
        floor_sd_s = "undefined" if s.floor_sd is None else _fmt(s.floor_sd)
        excess_s = "undefined" if s.excess is None else _fmt(s.excess)
        lines.append(
            f"| {s.method} | {s.m} | {drawn_s} | {s.n_seeds} | {mean_s} | {se_s} | {ci_s} | "
            f"{floor_mean_s} | {floor_sd_s} | {excess_s} |"
        )
    lines.append("")

    capped_rows = [s for s in label_stats if s.reused_from_m is not None]
    if capped_rows:
        capped_by_source: dict[int, list[int]] = {}
        for s in capped_rows:
            capped_by_source.setdefault(s.reused_from_m, []).append(s.m)
        cap_sentences = []
        for source_m in sorted(capped_by_source):
            capped_ms = sorted(set(capped_by_source[source_m]))
            cap_sentences.append(
                f"The rows at m equal to {_join_and([str(m) for m in capped_ms])} "
                f"repeat the m equal to {source_m} row exactly, statistics included, "
                f"because the label budget pool holds {source_m} labels per class at "
                "every larger m."
            )
        lines.append(" ".join(cap_sentences))
        lines.append("")

    below_all = all(f.below_floor_mean for f in floor_check)
    below_methods = [f.method for f in floor_check if f.below_floor_mean]
    above_methods = [f.method for f in floor_check if not f.below_floor_mean]
    exclude_one = [d.method for d in decay if not d.ratio_flag_at_or_below_one]
    include_one = [d.method for d in decay if d.ratio_flag_at_or_below_one]

    para: list[str] = []
    if below_all:
        para.append(
            "At K equal to 100 the 95 percent interval of every non oracle method sits "
            "entirely below the floor mean."
        )
    else:
        if below_methods:
            para.append(
                "At K equal to 100 the 95 percent interval sits entirely below the floor "
                f"mean for {', '.join(below_methods)}."
            )
        if above_methods:
            para.append(
                f"The interval reaches the floor mean or above it for {', '.join(above_methods)}."
            )

    if exclude_one:
        para.append(
            "The decay ratio interval excludes one for "
            f"{', '.join(exclude_one)}, so each of those methods decays faster than the "
            "oracle at the 95 percent level."
        )
    if include_one:
        para.append(
            f"The decay ratio interval reaches one or below for {', '.join(include_one)}."
        )

    if smallest_ppi is None:
        para.append(
            "The bootstrap interval of PPI's worst-class mean never contains the floor "
            "mean at any measured target label budget per class."
        )
    else:
        para.append(
            "The smallest target label budget per class at which PPI's bootstrap interval "
            f"contains the floor mean is {smallest_ppi}."
        )
    if smallest_oracle is None:
        para.append(
            "The bootstrap interval of the oracle's worst-class mean never contains the "
            "floor mean at any measured target label budget per class."
        )
    else:
        para.append(
            "The smallest target label budget per class at which the oracle's bootstrap "
            f"interval contains the floor mean is {smallest_oracle}."
        )

    if capped_rows:
        for source_m in sorted(capped_by_source):
            capped_ms = sorted(set(capped_by_source[source_m]))
            para.append(
                f"The label budget grid caps at {source_m} labels per class, so the m "
                f"equal to {_join_and([str(m) for m in capped_ms])} rows repeat the m "
                f"equal to {source_m} measurement."
            )

    if label_pairs:
        last = label_pairs[-1]
        if last.reused_from_m is not None:
            para.append(
                f"At the m equal to {last.m} point of the label budget grid, which "
                f"repeats the m equal to {last.reused_from_m} point exactly because the "
                f"pool caps at {last.reused_from_m} labels per class, PPI's worst-class "
                f"mean minus the oracle's paired worst-class mean is "
                f"{_fmt(last.diff_mean)}, with a 95 percent interval of "
                f"{_fmt_ci(last.diff_ci_lo, last.diff_ci_hi)}."
            )
        else:
            para.append(
                f"At {last.m} target labels per class, PPI's worst-class mean minus the "
                f"oracle's paired worst-class mean is {_fmt(last.diff_mean)}, with a 95 percent "
                f"interval of {_fmt_ci(last.diff_ci_lo, last.diff_ci_hi)}."
            )

    lines.append(" ".join(para))
    lines.append("")
    return "\n".join(lines)


def run_all(analysis_dir: Path, B: int, seed: int) -> tuple[dict, str]:
    controlled_records = {c: load_controlled_records(analysis_dir, c) for c in MODERATE_CACHES}
    idx = index_controlled_records(controlled_records)
    floored_records = {c: load_controlled_records(analysis_dir, c) for c in FLOORED_CACHES}
    label_payload = load_label_sweep(analysis_dir, LABEL_SWEEP_CACHE)

    rng = np.random.default_rng(seed)
    per_k = controlled_per_k(idx, MODERATE_CACHES, CONTROLLED_METHODS, K_VALUES, N_SUBSET_SEEDS, rng, B)
    decay, ratio_se_caveat = controlled_decay(idx, MODERATE_CACHES, CONTROLLED_METHODS, N_SUBSET_SEEDS, rng, B)
    vs_oracle = controlled_vs_oracle(idx, MODERATE_CACHES, NON_ORACLE_METHODS, K_VALUES, N_SUBSET_SEEDS, rng, B)
    floor_check = controlled_floor_check(per_k, k=100)
    floored = floored_cache_stats(floored_records, FLOORED_CACHES, rng, B)
    label_stats, label_pairs, smallest_ppi, smallest_oracle = label_sweep_stats(
        label_payload, LABEL_SWEEP_METHODS, rng, B
    )

    payload = to_json_payload(
        B=B, seed=seed, moderate_caches=MODERATE_CACHES, floored_caches=FLOORED_CACHES,
        label_sweep_cache=LABEL_SWEEP_CACHE, per_k=per_k, decay=decay, vs_oracle=vs_oracle,
        floor_check=floor_check, floored=floored, label_stats=label_stats, label_pairs=label_pairs,
        smallest_ppi=smallest_ppi, smallest_oracle=smallest_oracle, ratio_se_caveat=ratio_se_caveat,
    )
    md = render_markdown(
        B=B, per_k=per_k, decay=decay, vs_oracle=vs_oracle, floor_check=floor_check,
        floored=floored, label_stats=label_stats, label_pairs=label_pairs,
        smallest_ppi=smallest_ppi, smallest_oracle=smallest_oracle,
        moderate_caches=MODERATE_CACHES, floored_caches=FLOORED_CACHES,
        ratio_se_caveat=ratio_se_caveat,
    )
    return payload, md


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--out-json", type=Path, default=Path("outputs/analysis/sweep_uncertainty.json"))
    p.add_argument("--out-md", type=Path, default=Path("paper/camera_ready/SWEEP_UNCERTAINTY.md"))
    p.add_argument("--B", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload, md = run_all(args.analysis_dir, args.B, args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
