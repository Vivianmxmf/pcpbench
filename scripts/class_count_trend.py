#!/usr/bin/env python3
"""Class-count trend of worst-class coverage against the matched finite-sample reference."""

from __future__ import annotations

import argparse
import functools
import glob
import gzip
import json
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
from scipy import stats

from pcpbench.metrics.validity_floor import expected_min_coverage
from pcpbench.runner import METHODS


K_VALUES: tuple[int, ...] = (10, 25, 50, 100)
N_CLASSES = 100
N_CAL_PER_CLASS = 50
N_EVAL_PER_CLASS = 50
N_POOL_PER_CLASS = 50
ALPHA = 0.10
EVAL_FRAC = 0.5

CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise", "shot_noise", "defocus_blur", "gaussian_blur",
    "contrast", "brightness", "fog", "jpeg_compression",
)
SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)
BACKBONES: tuple[str, ...] = ("ViT-B/32", "ViT-L/14")
ORACLE = "oracle"
MODERATE_BACKBONE = "ViT-B/32"
MODERATE_CACHES: tuple[tuple[str, int], ...] = (
    ("contrast", 1), ("contrast", 2), ("contrast", 3), ("brightness", 1), ("brightness", 3),
)
MODERATE_SCOPE = "moderate five (ViT-B/32)"

Z_BOUND = 2
PRIMARY_PAIR: tuple[int, int] = (10, 100)
CI_LEVEL = 0.95
FAMILY_ALPHA = 0.05
FAMILY_METHODS: tuple[str, ...] = ("split", "mondrian", "weighted", "ppi", "acp")
FAMILY_TERMS: tuple[str, ...] = ("same_class", "selection")
FAMILY_SIZE = len(FAMILY_METHODS) * len(BACKBONES) * len(FAMILY_TERMS)
assert FAMILY_SIZE == 20
BONFERRONI_LEVEL = 1.0 - FAMILY_ALPHA / FAMILY_SIZE
BOUND_SHARE = 0.5
DEFAULT_REPS = 20_000
_ZERO_TOL = 1e-12

ACP = "acp"
ACP_TIE_BREAK = "random"
ACP_TIE_BREAK_UNRECORDED = "index"
NO_SHIFT_BASE: tuple[str, int] = ("contrast", 3)
LIBRARY_KEYS: tuple[str, ...] = ("python", "numpy", "sklearn")
MARGINAL_METHODS: tuple[str, ...] = ("split", "weighted", "clustered", "acp")

_CACHE_RE = re.compile(r"^cifar100c_(?P<vitl>vitL_)?(?P<corruption>[a-z_]+)_s(?P<severity>[1-5])$")


class CacheKey(NamedTuple):
    backbone: str
    corruption: str
    severity: int


class Reference(NamedTuple):
    mean: float
    sd: float


@dataclass(frozen=True)
class Sweep:
    """One verified sweep file, indexed for the analysis."""

    name: str
    key: CacheKey
    config: dict
    n_seeds: int
    cells: dict[tuple[str, int, int], dict]
    class_orders: tuple[tuple[int, ...], ...] | None
    provenance: dict


def cache_key(cache_path: str) -> CacheKey:
    """Backbone, corruption and severity from a cache path of the CIFAR-100-C grid."""
    stem = Path(cache_path).name
    if stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    match = _CACHE_RE.match(stem)
    if match is None or match["corruption"] not in CORRUPTIONS:
        raise ValueError(f"{cache_path!r} is not a cache of the CIFAR-100-C severity grid")
    backbone = "ViT-L/14" if match["vitl"] else "ViT-B/32"
    return CacheKey(backbone, match["corruption"], int(match["severity"]))


def cache_label(key: CacheKey) -> str:
    """The name the sweep files use: no dataset prefix, ``vitL_`` for ViT-L/14."""
    prefix = "vitL_" if key.backbone == "ViT-L/14" else ""
    return f"{prefix}{key.corruption}_s{key.severity}"


def load_payload(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _is_coverage(value: object) -> bool:
    return _is_finite_number(value) and 0.0 <= value <= 1.0


def verify_payload(
    payload: dict, name: str, *, design: str, view: str, n_seeds: int | None = None,
    methods: Sequence[str] | None = None, k_values: Sequence[int] = K_VALUES,
) -> list[str]:
    """Every reason this file cannot enter the analysis; empty when there is none."""
    methods = tuple(sorted(METHODS)) if methods is None else tuple(methods)
    problems: list[str] = []

    def need(ok: bool, message: str) -> None:
        if not ok:
            problems.append(f"{name}: {message}")

    config = payload.get("config")
    if not isinstance(config, dict):
        return [f"{name}: no config block"]
    found_design = config.get("subset_design", "independent")
    need(found_design == design, f"subset design is {found_design!r}, expected {design!r}")
    found_view = config.get("view", "shift")
    need(found_view == view, f"view is {found_view!r}, expected {view!r}")
    need(config.get("alpha") == ALPHA, f"alpha is {config.get('alpha')!r}, expected {ALPHA}")
    need(config.get("n_cal_per_class") == N_CAL_PER_CLASS,
         f"n_cal_per_class is {config.get('n_cal_per_class')!r}, expected {N_CAL_PER_CLASS}")
    need(config.get("eval_frac") == EVAL_FRAC,
         f"eval_frac is {config.get('eval_frac')!r}, expected {EVAL_FRAC}")
    need(list(config.get("k_values") or []) == list(k_values),
         f"k_values is {config.get('k_values')!r}, expected {list(k_values)}")
    seeds = config.get("subset_seeds")
    if not isinstance(seeds, int) or seeds < 1:
        return problems + [f"{name}: subset_seeds is {seeds!r}"]
    if n_seeds is not None:
        need(seeds == n_seeds, f"subset_seeds is {seeds}, expected {n_seeds}")
    try:
        cache_key(str(config.get("cache", "")))
    except ValueError as exc:
        problems.append(f"{name}: cache: {exc}")
    if view == "no_shift":
        need(isinstance(config.get("view_seed"), int), "view_seed is missing for a no-shift view")

    expected = {(m, k, s) for m in methods for k in k_values for s in range(seeds)}
    seen: dict[tuple, dict] = {}
    duplicates = 0
    for record in payload.get("records") or []:
        cell = (record.get("method"), record.get("k"), record.get("subset_seed"))
        if cell in seen:
            duplicates += 1
        seen[cell] = record
    need(duplicates == 0, f"{duplicates} duplicate record(s)")
    missing = sorted(expected - set(seen))
    need(not missing, f"{len(missing)} record(s) missing, the first is {missing[:1]}")
    unexpected = sorted(set(seen) - expected, key=str)
    need(not unexpected, f"{len(unexpected)} record(s) outside the design, the first is {unexpected[:1]}")

    bad: dict[str, int] = {}
    for cell in sorted(expected & set(seen)):
        record, k = seen[cell], cell[1]
        if record.get("n_shortfall_classes") != 0:
            bad["a calibration shortfall"] = bad.get("a calibration shortfall", 0) + 1
        if record.get("n_cal_total") != N_CAL_PER_CLASS * k:
            bad["n_cal_total off 50 per class"] = bad.get("n_cal_total off 50 per class", 0) + 1
        if record.get("n_eval_total") != N_EVAL_PER_CLASS * k:
            bad["n_eval_total off 50 per class"] = bad.get("n_eval_total off 50 per class", 0) + 1
        if record.get("n_target_cal_pool") not in (0, N_POOL_PER_CLASS * k):
            bad["n_target_cal_pool off 50 per class"] = bad.get(
                "n_target_cal_pool off 50 per class", 0) + 1
        per_class = record.get("per_class_coverage")
        if (not isinstance(per_class, list) or len(per_class) != k
                or any(not _is_coverage(v) for v in per_class)):
            bad["per_class_coverage not K coverages in [0, 1]"] = bad.get(
                "per_class_coverage not K coverages in [0, 1]", 0) + 1
        elif not (_is_coverage(record.get("worst_class"))
                  and math.isclose(min(per_class), record["worst_class"], abs_tol=1e-9)):
            bad["worst_class not the minimum of per_class_coverage"] = bad.get(
                "worst_class not the minimum of per_class_coverage", 0) + 1
        if not _is_finite_number(record.get("set_over_k")):
            bad["set_over_k not a finite number"] = bad.get("set_over_k not a finite number", 0) + 1
    for reason, count in bad.items():
        problems.append(f"{name}: {count} record(s) with {reason}")

    if design == "nested":
        if any(cell[0] == ACP for cell in seen):
            found_rule = config.get("acp_tie_break")
            need(found_rule == ACP_TIE_BREAK,
                 f"acp_tie_break is {found_rule!r}, expected {ACP_TIE_BREAK!r}: the file holds "
                 "ACP records that were not written with the random tie order")
        orders = payload.get("class_orders")
        orders_ok = (isinstance(orders, list) and len(orders) == seeds
                     and all(isinstance(o, list) and sorted(o) == list(range(N_CLASSES))
                             for o in orders))
        need(orders_ok, f"class_orders is not {seeds} orders of the {N_CLASSES} classes")
        need(config.get("method_seed") == 0,
             f"method_seed is {config.get('method_seed')!r}, expected 0")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            problems.append(f"{name}: no provenance block")
        else:
            version = provenance.get("code_version")
            need(isinstance(version, str) and version not in ("", "unknown"),
                 f"provenance.code_version is {version!r}, not a recorded code version")
            need(len(str(provenance.get("source_cal_sha256") or "")) == 64,
                 "provenance.source_cal_sha256 is not a SHA-256 digest")
            for library in LIBRARY_KEYS:
                found_version = provenance.get(library)
                need(isinstance(found_version, str) and found_version != "",
                     f"provenance.{library} is {found_version!r}, not a recorded version")
        if orders_ok:
            problems += [f"{name}: {message}" for message in _nested_detail_problems(
                payload, seen, orders, seeds, k_values)]
    return problems


def _nested_detail_problems(payload: dict, seen: dict[tuple, dict], orders: list[list[int]],
                            seeds: int, k_values: Sequence[int]) -> list[str]:
    out: list[str] = []
    facts: dict[tuple, dict] = {}
    for fact in payload.get("problems") or []:
        facts[(fact.get("k"), fact.get("subset_seed"))] = fact
    wanted = {(k, s) for k in k_values for s in range(seeds)}
    if set(facts) != wanted:
        out.append(f"problems block holds {len(set(facts) & wanted)} of the {len(wanted)} "
                   "(K, seed) entries")
    wrong_layout = wrong_eval = wrong_top1 = 0
    for (k, s) in sorted(set(facts) & wanted):
        fact = facts[(k, s)]
        if fact.get("class_ids") != sorted(orders[s][:k]):
            wrong_layout += 1
        if fact.get("n_eval_per_class") != [N_EVAL_PER_CLASS] * k:
            wrong_eval += 1
        top1 = fact.get("eval_top1_per_class")
        if (not isinstance(top1, list) or len(top1) != k
                or any(not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0 for v in top1)):
            wrong_top1 += 1
    if wrong_layout:
        out.append(f"{wrong_layout} problem(s) whose class_ids are not the sorted first K "
                   "classes of the seed's order")
    if wrong_eval:
        out.append(f"{wrong_eval} problem(s) without {N_EVAL_PER_CLASS} evaluation rows for "
                   "every class")
    if wrong_top1:
        out.append(f"{wrong_top1} problem(s) whose eval_top1_per_class is not K accuracies")
    if not (wrong_layout or wrong_top1) and set(facts) == wanted:
        rising = _rising_top1(facts, seeds, k_values)
        if rising:
            out.append(f"{rising} class(es) whose top-1 accuracy rises with K: a class keeps "
                       "its evaluation rows and only gains competitors, so the per-class "
                       "vectors are not laid out by class_ids")
    no_counts = off_counts = 0
    for (method, k, _), record in sorted(seen.items(), key=str):
        counts = record.get("n_cal_per_class")
        if not isinstance(counts, list) or len(counts) != k:
            no_counts += 1
        elif method in ("mondrian", ORACLE) and counts != [N_CAL_PER_CLASS] * k:
            off_counts += 1
    if no_counts:
        out.append(f"{no_counts} record(s) without n_cal_per_class for every class")
    if off_counts:
        out.append(f"{off_counts} mondrian or oracle record(s) without {N_CAL_PER_CLASS} "
                   "calibration points for every class")
    return out


def _rising_top1(facts: dict[tuple, dict], seeds: int, k_values: Sequence[int]) -> int:
    ordered = sorted(k_values)
    rising = 0
    for seed in range(seeds):
        tables = {k: dict(zip(facts[(k, seed)]["class_ids"],
                              facts[(k, seed)]["eval_top1_per_class"])) for k in ordered}
        for small, large in zip(ordered, ordered[1:]):
            rising += sum(1 for g, accuracy in tables[small].items()
                          if tables[large].get(g, 0.0) > accuracy + 1e-12)
    return rising


def set_problems(sweeps: Sequence[Sweep], views: Sequence[Sweep], design: str) -> list[str]:
    """Reasons the verified files do not form one run."""
    if design != "nested":
        return []
    out: list[str] = []
    everything = [*sweeps, *views]
    versions = sorted({str(s.provenance.get("code_version")) for s in everything})
    if len(versions) > 1:
        out.append(f"the files were written by {len(versions)} code versions "
                   f"({', '.join(versions)}); one run writes one")
    for library in LIBRARY_KEYS:
        found = sorted({str(s.provenance.get(library)) for s in everything})
        if len(found) > 1:
            out.append(f"the files were written under {len(found)} versions of {library} "
                       f"({', '.join(found)}); one run writes one")
    base_label = f"{NO_SHIFT_BASE[0]}_s{NO_SHIFT_BASE[1]}"
    counts = {}
    for backbone in BACKBONES:
        mine = [v for v in views if v.key.backbone == backbone]
        counts[backbone] = len(mine)
        seeds = [v.config.get("view_seed") for v in mine]
        if len(set(seeds)) != len(seeds):
            out.append(f"{backbone}: the no-shift files repeat a view seed ({sorted(seeds)})")
        bases = sorted({cache_label(v.key) for v in mine})
        if len(bases) > 1:
            out.append(f"{backbone}: the no-shift files come from {len(bases)} caches "
                       f"({', '.join(bases)})")
        off_base = sorted({cache_label(v.key) for v in mine
                           if (v.key.corruption, v.key.severity) != NO_SHIFT_BASE})
        if off_base:
            out.append(f"{backbone}: a no-shift file is a view of {', '.join(off_base)}; the "
                       f"no-shift view is drawn from {base_label} for both backbones")
    if len(set(counts.values())) > 1:
        split = ", ".join(f"{count} of {backbone}" for backbone, count in counts.items())
        out.append(f"the no-shift files are not the same number for each backbone ({split})")
    return out


def as_sweep(name: str, payload: dict) -> Sweep:
    """Index a payload that passed ``verify_payload``."""
    config = payload["config"]
    orders = payload.get("class_orders")
    return Sweep(
        name=name,
        key=cache_key(config["cache"]),
        config=config,
        n_seeds=int(config["subset_seeds"]),
        cells={(r["method"], r["k"], r["subset_seed"]): r for r in payload["records"]},
        class_orders=None if orders is None else tuple(tuple(o) for o in orders),
        provenance=payload.get("provenance") or {},
    )


def load_sweeps(
    paths: Iterable[Path], *, design: str, view: str, n_seeds: int | None = None,
) -> tuple[list[Sweep], list[str]]:
    """Read and verify every file."""
    sweeps: list[Sweep] = []
    problems: list[str] = []
    for path in paths:
        try:
            payload = load_payload(path)
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            problems.append(f"{path.name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        found = verify_payload(payload, path.name, design=design, view=view, n_seeds=n_seeds)
        problems += found
        if not found:
            sweeps.append(as_sweep(path.name, payload))
    return sweeps, problems


def grid_problems(sweeps: Sequence[Sweep], backbones: Sequence[str] = BACKBONES) -> list[str]:
    """Each backbone needs its 40 caches once, all with one seed count."""
    problems: list[str] = []
    by_key: dict[CacheKey, int] = {}
    for sweep in sweeps:
        by_key[sweep.key] = by_key.get(sweep.key, 0) + 1
    for key, count in sorted(by_key.items()):
        if count > 1:
            problems.append(f"{cache_label(key)}: more than one file ({count}) for this cache")
        if key.backbone not in backbones:
            problems.append(f"{cache_label(key)}: backbone {key.backbone} was not asked for")
    for backbone in backbones:
        missing = [cache_label(CacheKey(backbone, c, s)) for c in CORRUPTIONS for s in SEVERITIES
                   if CacheKey(backbone, c, s) not in by_key]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            problems.append(f"{backbone}: {len(missing)} of 40 caches missing ({shown})")
    seed_counts = sorted({sweep.n_seeds for sweep in sweeps})
    if len(seed_counts) > 1:
        problems.append(f"the files differ in subset_seeds: {seed_counts}")
    return problems


def reference_table(k_values: Sequence[int], reps: int = DEFAULT_REPS) -> dict[int, Reference]:
    """Matched reference at every K: mean and single-draw sd of the worst class of an exactly
    valid per-class method with 50 calibration and 50 evaluation points."""
    table = {}
    for k in k_values:
        floor = expected_min_coverage(k, N_CAL_PER_CLASS, N_EVAL_PER_CLASS, alpha=ALPHA, reps=reps)
        table[k] = Reference(float(floor.mean), float(floor.sd))
    return table


def seed_interval(seed_means: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided t interval for the mean of per-seed values."""
    values = np.asarray(seed_means, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("an interval needs the values of at least two subset seeds")
    half = float(stats.t.ppf(0.5 + level / 2.0, values.size - 1)
                 * values.std(ddof=1) / math.sqrt(values.size))
    centre = float(values.mean())
    return centre - half, centre + half


def _coverage_interval(seed_means: np.ndarray) -> tuple[float, float]:
    lo, hi = seed_interval(seed_means, CI_LEVEL)
    return max(lo, 0.0), min(hi, 1.0)


def band_counts(cache_means: np.ndarray, reference_mean: float, half_width: float) -> dict:
    below = int(np.sum(cache_means < reference_mean - half_width))
    above = int(np.sum(cache_means > reference_mean + half_width))
    return {"below": below, "within": int(cache_means.size) - below - above, "above": above,
            "half_width": float(half_width)}


def _zero_share(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values) <= _ZERO_TOL))


def _at(values: np.ndarray, limit: float) -> np.ndarray:
    return np.abs(np.asarray(values, dtype=float) - limit) <= _ZERO_TOL


@functools.lru_cache(maxsize=None)
def _subset_min_weights(n: int, m: int) -> np.ndarray:
    total = math.comb(n, m)
    return np.array([math.comb(n - i, m - 1) / total for i in range(1, n - m + 2)])


def expected_subset_min(values: np.ndarray, m: int) -> float:
    """Exact mean, over every subset of ``m`` of the values, of the subset's minimum."""
    ordered = np.sort(np.asarray(values, dtype=float).ravel())
    if not 1 <= m <= ordered.size:
        raise ValueError(f"a subset of {m} cannot be drawn from {ordered.size} values")
    weights = _subset_min_weights(int(ordered.size), int(m))
    return float(np.dot(ordered[: weights.size], weights))


def skewness(values: np.ndarray) -> float | None:
    """Adjusted sample skewness of per-seed values; None when it is not defined (fewer than three
    values, or no spread)."""
    data = np.asarray(values, dtype=float).ravel()
    if data.size < 3 or float(np.ptp(data)) <= _ZERO_TOL:
        return None
    return float(stats.skew(data, bias=False))


def _excluded_sign(lo: float, hi: float) -> int:
    if hi < -_ZERO_TOL:
        return -1
    if lo > _ZERO_TOL:
        return 1
    return 0


class Scope(NamedTuple):
    """A fixed set of caches of one backbone."""

    name: str
    backbone: str
    keys: tuple[CacheKey, ...]
    by_family: bool


def analysis_scopes() -> tuple[Scope, ...]:
    grids = [Scope(b, b, tuple(CacheKey(b, c, s) for c in CORRUPTIONS for s in SEVERITIES), True)
             for b in BACKBONES]
    moderate = Scope(MODERATE_SCOPE, MODERATE_BACKBONE,
                     tuple(CacheKey(MODERATE_BACKBONE, c, s) for c, s in MODERATE_CACHES), False)
    return (*grids, moderate)


def cell_matrix(by_key: dict[CacheKey, Sweep], keys: Sequence[CacheKey], method: str, k: int,
                field: str) -> np.ndarray:
    """(cache, seed) array of one record field."""
    n_seeds = by_key[keys[0]].n_seeds
    return np.array([[by_key[key].cells[(method, k, seed)][field] for seed in range(n_seeds)]
                     for key in keys], dtype=float)


def _family_means(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(len(CORRUPTIONS), len(SEVERITIES), -1).mean(axis=(1, 2))


def class_coverage(sweep: Sweep, method: str, k: int, seed: int) -> np.ndarray:
    """Per-class coverage laid out by ORIGINAL class id, NaN for a class outside the K-subset."""
    ids = np.sort(np.asarray(sweep.class_orders[seed][:k]))
    out = np.full(N_CLASSES, np.nan)
    out[ids] = sweep.cells[(method, k, seed)]["per_class_coverage"]
    return out


class Terms(NamedTuple):
    """One (cache, seed) cell of the decomposition."""

    same_class: float
    selection: float
    total: float
    mean_class_change: float
    worst_small: float
    kept_min_large: float
    worst_large: float
    selection_rb: float


_T = {name: index for index, name in enumerate(Terms._fields)}


def decomposition_terms(
    sweep: Sweep, method: str, seed: int, k_small: int, k_large: int,
    reference: dict[int, Reference],
) -> Terms:
    """The decomposition of one seed of one file."""
    small = class_coverage(sweep, method, k_small, seed)
    large = class_coverage(sweep, method, k_large, seed)
    kept = np.asarray(sweep.class_orders[seed][:k_small])
    present = large[~np.isnan(large)]
    worst_small = float(np.min(small[kept]))
    kept_min_large = float(np.min(large[kept]))
    worst_large = float(np.min(present))
    reference_step = reference[k_large].mean - reference[k_small].mean
    return Terms(
        same_class=kept_min_large - worst_small,
        selection=worst_large - kept_min_large - reference_step,
        total=worst_large - worst_small - reference_step,
        mean_class_change=float(np.mean(large[kept] - small[kept])),
        worst_small=worst_small,
        kept_min_large=kept_min_large,
        worst_large=worst_large,
        selection_rb=worst_large - expected_subset_min(present, k_small) - reference_step,
    )


def term_array(sweeps: Sequence[Sweep], method: str,
               reference: dict[int, Reference]) -> np.ndarray:
    """(file, seed, field) array of ``Terms`` for the primary pair."""
    n_seeds = sweeps[0].n_seeds
    return np.array([[decomposition_terms(sweep, method, seed, *PRIMARY_PAIR, reference)
                      for seed in range(n_seeds)] for sweep in sweeps], dtype=float)


def _per_k(by_key: dict[CacheKey, Sweep], scope: Scope, methods: Sequence[str], design: str,
           reference: dict[int, Reference]) -> list[dict]:
    rows = []
    for method in methods:
        for k in K_VALUES:
            matrix = cell_matrix(by_key, scope.keys, method, k, "worst_class")
            lo, hi = _coverage_interval(matrix.mean(axis=0))
            ref = reference[k]
            row = {"method": method, "k": k, "mean": float(matrix.mean()), "ci_lo": lo,
                   "ci_hi": hi, "reference": ref.mean, "diff": float(matrix.mean() - ref.mean),
                   "zero_share": _zero_share(matrix),
                   "set_over_k": float(cell_matrix(by_key, scope.keys, method, k,
                                                   "set_over_k").mean())}
            if scope.by_family:
                families = _family_means(matrix)
                row["family_min"], row["family_max"] = float(families.min()), float(families.max())
                if design == "independent":
                    cache_means = matrix.mean(axis=1)
                    row["band_single_draw"] = band_counts(cache_means, ref.mean, Z_BOUND * ref.sd)
                    row["band_seed_mean"] = band_counts(
                        cache_means, ref.mean, Z_BOUND * ref.sd / math.sqrt(matrix.shape[1]))
            rows.append(row)
    return rows


def _interval_stats(grid: np.ndarray, *, in_family: bool) -> dict:
    seed_means = grid.mean(axis=0)
    lo, hi = seed_interval(seed_means, CI_LEVEL)
    stat = {"mean": float(grid.mean()), "ci_lo": lo, "ci_hi": hi,
            "bonferroni_lo": None, "bonferroni_hi": None, "excludes_zero": None,
            "skewness": skewness(seed_means)}
    if in_family:
        wide_lo, wide_hi = seed_interval(seed_means, BONFERRONI_LEVEL)
        stat.update({"bonferroni_lo": wide_lo, "bonferroni_hi": wide_hi,
                     "excludes_zero": _excluded_sign(wide_lo, wide_hi)})
    return stat


def _bound_stats(grid: np.ndarray, bound: np.ndarray) -> dict:
    free = grid[~bound]
    share = float(bound.mean())
    return {"bound_share": share, "n_free": int(free.size),
            "free_mean": float(free.mean()) if free.size else None,
            "marked": bool(share >= BOUND_SHARE)}


def _decomposition_row(
    method: str, terms: np.ndarray, *, in_family: bool, by_family: bool,
    reference_step: float, unit: str = "caches", no_shift_terms: np.ndarray | None = None,
) -> dict:
    k_small, k_large = PRIMARY_PAIR
    worst_small = terms[:, :, _T["worst_small"]]
    kept_min_large = terms[:, :, _T["kept_min_large"]]
    worst_large = terms[:, :, _T["worst_large"]]
    bound = {
        "same_class": _at(worst_small, 0.0) | _at(worst_small, 1.0),
        "selection": _at(kept_min_large, 0.0) | _at(worst_large, 1.0),
    }
    row = {"method": method, "k_small": k_small, "k_large": k_large, "in_family": in_family,
           "zero_share_k_small": _zero_share(worst_small),
           "zero_share_k_large": _zero_share(worst_large),
           "limits": {name: {"at_zero": float(_at(values, 0.0).mean()),
                             "at_one": float(_at(values, 1.0).mean())}
                      for name, values in (("worst_small", worst_small),
                                           ("kept_min_large", kept_min_large),
                                           ("worst_large", worst_large))}}
    for name in FAMILY_TERMS:
        grid = terms[:, :, _T[name]]
        stat = _interval_stats(grid, in_family=in_family)
        stat[f"n_{unit}"] = int(grid.shape[0])
        stat[f"n_{unit}_negative"] = int(np.sum(grid.mean(axis=1) < -_ZERO_TOL))
        stat[f"{unit}_means"] = [float(value) for value in grid.mean(axis=1)]
        if by_family:
            families = _family_means(grid)
            stat.update({"n_families_negative": int(np.sum(families < -_ZERO_TOL)),
                         "family_min": float(families.min()),
                         "family_max": float(families.max())})
        stat.update(_bound_stats(grid, bound[name]))
        if no_shift_terms is not None:
            other = no_shift_terms[:, :, _T[name]]
            other_lo, other_hi = seed_interval(other.mean(axis=0), CI_LEVEL)
            paired = grid.mean(axis=0) - other.mean(axis=0)
            paired_lo, paired_hi = seed_interval(paired, CI_LEVEL)
            stat["no_shift"] = {"mean": float(other.mean()), "ci_lo": other_lo, "ci_hi": other_hi}
            stat["minus_no_shift"] = {"mean": float(paired.mean()), "ci_lo": paired_lo,
                                      "ci_hi": paired_hi}
        row[name] = stat
    row["selection"]["pinned_part"] = float(row["selection"]["bound_share"] * -reference_step)
    row["selection_rb"] = _interval_stats(terms[:, :, _T["selection_rb"]], in_family=in_family)
    excluded = row["selection"]["excludes_zero"]
    row["selection"]["rb_reproduces"] = (
        None if not excluded else bool(row["selection_rb"]["excludes_zero"] == excluded))
    row["total"] = float(terms[:, :, _T["total"]].mean())
    row["mean_class_change"] = float(terms[:, :, _T["mean_class_change"]].mean())
    return row


READINGS = ("outside", "bound", "contains", "withheld", "negative", "positive")


def term_reading(row: dict, term: str) -> str:
    """The reading of one term (``same_class`` or ``selection``) of a decomposition row."""
    if term not in FAMILY_TERMS:
        raise ValueError(f"{term!r} is not a term of the decomposition")
    stat = row[term]
    if not row["in_family"]:
        return "outside"
    if stat["marked"]:
        return "bound"
    sign = stat["excludes_zero"]
    if sign is None:
        raise ValueError("a term of a family row has no Bonferroni interval")
    if sign == 0:
        return "contains"
    free_mean = stat["free_mean"]
    if free_mean is not None and free_mean * sign < -_ZERO_TOL:
        return "withheld"
    if term == "selection" and stat["rb_reproduces"] is False:
        return "withheld"
    return "negative" if sign < 0 else "positive"


def _decomposition(by_key: dict[CacheKey, Sweep], scope: Scope, methods: Sequence[str],
                   reference: dict[int, Reference],
                   no_shift_terms: dict[str, np.ndarray] | None = None) -> list[dict]:
    step = reference[PRIMARY_PAIR[1]].mean - reference[PRIMARY_PAIR[0]].mean
    sweeps = [by_key[key] for key in scope.keys]
    return [
        _decomposition_row(
            method, term_array(sweeps, method, reference),
            in_family=scope.by_family and method in FAMILY_METHODS, by_family=scope.by_family,
            reference_step=step,
            no_shift_terms=None if no_shift_terms is None else no_shift_terms[method])
        for method in methods
    ]


def _scope_block(by_key: dict[CacheKey, Sweep], scope: Scope, methods: Sequence[str],
                 design: str, reference: dict[int, Reference],
                 no_shift_terms: dict[str, np.ndarray] | None = None) -> dict:
    return {
        "backbone": scope.backbone, "n_caches": len(scope.keys),
        "caches": [cache_label(key) for key in scope.keys],
        "selected_on_outcomes": not scope.by_family,
        "per_k": _per_k(by_key, scope, methods, design, reference),
        "decomposition": (_decomposition(by_key, scope, methods, reference, no_shift_terms)
                          if design == "nested" else None),
    }


def _moderate_caches(by_key: dict[CacheKey, Sweep], methods: Sequence[str]) -> list[dict]:
    out = []
    for corruption, severity in MODERATE_CACHES:
        key = CacheKey(MODERATE_BACKBONE, corruption, severity)
        rows = []
        for method in methods:
            for k in K_VALUES:
                values = cell_matrix(by_key, (key,), method, k, "worst_class")
                rows.append({"method": method, "k": k, "mean": float(values.mean()),
                             "zero_share": _zero_share(values)})
        out.append({"cache": cache_label(key), "per_k": rows})
    return out


def _no_shift(views: Sequence[Sweep], by_key: dict[CacheKey, Sweep], methods: Sequence[str],
              reference: dict[int, Reference], terms: dict[str, np.ndarray]) -> dict:
    base = views[0].key
    n_seeds = views[0].n_seeds
    step = reference[PRIMARY_PAIR[1]].mean - reference[PRIMARY_PAIR[0]].mean
    decomposition = [
        _decomposition_row(method, terms[method], in_family=False, by_family=False,
                           reference_step=step, unit="views")
        for method in methods
    ]
    rows = []
    for method in methods:
        for k in K_VALUES:
            matrix = np.array([[view.cells[(method, k, seed)]["worst_class"]
                                for seed in range(n_seeds)] for view in views], dtype=float)
            lo, hi = _coverage_interval(matrix.mean(axis=0))
            sizes = [view.cells[(method, k, seed)]["set_over_k"]
                     for view in views for seed in range(n_seeds)]
            rows.append({"method": method, "k": k, "mean": float(matrix.mean()), "ci_lo": lo,
                         "ci_hi": hi, "reference": reference[k].mean,
                         "diff": float(matrix.mean() - reference[k].mean),
                         "zero_share": _zero_share(matrix), "set_over_k": float(np.mean(sizes)),
                         "base_cache_shift_mean": float(cell_matrix(
                             by_key, (base,), method, k, "worst_class").mean())})
    return {"n_views": len(views), "base_cache": cache_label(base),
            "view_seeds": sorted(v.config["view_seed"] for v in views), "per_k": rows,
            "decomposition": decomposition}


def analyse(
    sweeps: Sequence[Sweep], *, design: str, no_shift: Sequence[Sweep] | None = None,
    reps: int = DEFAULT_REPS,
) -> dict:
    """Every statistic of the class-count trend, as one JSON-ready dict."""
    problems = grid_problems(sweeps)
    if problems:
        raise ValueError("the sweep files do not form the grid: " + "; ".join(problems))
    if any(sweep.config.get("subset_design", "independent") != design for sweep in sweeps):
        raise ValueError(f"a sweep file is not of the {design} design")
    methods = tuple(sorted(METHODS))
    by_key = {sweep.key: sweep for sweep in sweeps}
    reference = reference_table(K_VALUES, reps)
    nested = design == "nested"

    views_of = {backbone: [view for view in no_shift or [] if view.key.backbone == backbone]
                for backbone in BACKBONES}
    no_shift_terms: dict[str, dict[str, np.ndarray]] = {}
    for backbone, views in views_of.items():
        if not views:
            continue
        if not nested:
            raise ValueError("the no-shift view is read with the nested design only")
        if any(view.n_seeds != sweeps[0].n_seeds for view in views):
            raise ValueError("a no-shift file differs from the sweep files in subset_seeds")
        no_shift_terms[backbone] = {method: term_array(views, method, reference)
                                    for method in methods}

    scopes = {scope.name: _scope_block(by_key, scope, methods, design, reference,
                                       no_shift_terms.get(scope.backbone))
              for scope in analysis_scopes()}
    no_shift_block = None
    if no_shift_terms:
        no_shift_block = {backbone: _no_shift(views_of[backbone], by_key, methods, reference,
                                              no_shift_terms[backbone])
                          for backbone in BACKBONES if backbone in no_shift_terms}

    return {
        "config": {
            "design": design, "n_files": len(sweeps), "n_seeds": sweeps[0].n_seeds,
            "backbones": list(BACKBONES), "k_values": list(K_VALUES), "alpha": ALPHA,
            "n_cal_per_class": N_CAL_PER_CLASS, "n_eval_per_class": N_EVAL_PER_CLASS,
            "reference_reps": reps, "band_z": Z_BOUND, "ci_level": CI_LEVEL,
            "interval": ("two-sided t interval on the per-seed means of a scope, unadjusted "
                         "outside the family; the caches of a scope are fixed conditions, so "
                         "it is conditional on the image set"),
            "family": ({"size": FAMILY_SIZE, "level": BONFERRONI_LEVEL,
                        "methods": list(FAMILY_METHODS), "terms": list(FAMILY_TERMS),
                        "scopes": list(BACKBONES), "k_pair": list(PRIMARY_PAIR),
                        "bound_share": BOUND_SHARE} if nested else None),
            "acp_tie_breaks": sorted({str(s.config.get("acp_tie_break", ACP_TIE_BREAK_UNRECORDED))
                                      for s in sweeps}),
            "code_versions": sorted({s.provenance.get("code_version") for s in sweeps
                                     if s.provenance.get("code_version")}),
            "distinct_source_sets": ({b: len({s.provenance.get("source_cal_sha256")
                                              for s in sweeps if s.key.backbone == b})
                                      for b in BACKBONES} if nested else None),
        },
        "reference": {str(k): {"mean": ref.mean, "sd": ref.sd} for k, ref in reference.items()},
        "scopes": scopes,
        "moderate_caches": _moderate_caches(by_key, methods),
        "no_shift": no_shift_block,
    }


_METHOD_NAMES = {"acp": "ACP", "clustered": "Clustered", "mondrian": "Source-Mondrian",
                 "oracle": "Oracle", "ppi": "PPI", "split": "Split", "weighted": "Weighted"}
_METHOD_ORDER = ("oracle", "mondrian", "clustered", "split", "weighted", "ppi", "acp")


def _f3(x: float) -> str:
    text = f"{x:.3f}"
    return "0.000" if text == "-0.000" else text


def _f2(x: float) -> str:
    text = f"{x:.2f}"
    return "0.00" if text == "-0.00" else text


def _signed(x: float) -> str:
    text = _f3(x)
    return text if text.startswith("-") or text == "0.000" else f"+{text}"


def _span(lo: float, hi: float) -> str:
    return f"[{_f3(lo)}, {_f3(hi)}]"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def _find(rows: Sequence[dict], **match) -> dict:
    return next(r for r in rows if all(r[key] == value for key, value in match.items()))


def _k_header(first: str) -> list[str]:
    return [first] + [f"K={k}" for k in K_VALUES]


def _reading_notes(result: dict) -> list[str]:
    notes = [
        "An interval is the two-sided t interval on the per-seed means of a scope. The subset "
        "seeds are the replicates. The caches of a scope are fixed conditions that share one "
        "image set and one backbone, so no interval speaks for other corruptions.",
        "The per-K intervals are unadjusted and conditional on the image set: the seeds "
        "redraw rows inside one fixed set of images, and the reference is not conditional on "
        "it. A simulation of exactly valid per-class calibration put the standard deviation "
        "over image sets at 0.005 for K 10 and 0.008 for K 100, so a per-K interval that "
        "misses the reference by less than about 0.02 is not read as evidence.",
        "The oracle sits on the reference by construction: a random pool and evaluation "
        "split of a class's rows is exchangeable whatever the shift.",
        "Split, weighted, clustered and ACP calibrate one threshold across classes and target "
        "marginal coverage, so they need not sit on a per-class reference even without shift.",
        "The moderate five caches were chosen from realised outcomes of the earlier "
        "independent run. Their scope is descriptive.",
    ]
    rules = result["config"]["acp_tie_breaks"]
    if rules != [ACP_TIE_BREAK]:
        notes.append(
            "ACP's values in these files come from the tie rule " + " and ".join(rules) + ". "
            "Under the index rule its audited probabilities, which tie every class below the "
            "top one, were ordered by class index, so its per-class coverage followed the "
            "class index. The rule is now random.")
    if result["config"]["design"] == "nested":
        notes += [
            "Write a for the worst-class value at K 10, b for the minimum over the K 10 "
            "classes of their coverage inside the K 100 problem, and c for the worst-class "
            "value at K 100. The selection term is pinned at minus the reference's decline "
            "when b is 0 or c is 1, whatever the method does, and the same-class term is "
            "one-sided when a is 0 or 1. A term marked bound has at least "
            f"{_f2(BOUND_SHARE)} of its cells in that state and is not to be interpreted. "
            "With a smaller share the mean over the other cells is read beside the term, and "
            "an exclusion of zero is not reported when that mean has the other sign.",
            "Every cache of a scope shares one class order per seed, which skews the per-seed "
            "means. The skewness is printed beside the intervals, and the selection term is "
            "repeated with b replaced by its expectation over a random K 10 subset. A family "
            "exclusion that this version does not reproduce is flagged and not reported as "
            "evidence.",
            "The no-shift terms are the yardstick for the methods that target marginal "
            "coverage: a negative term of such a method under shift is a shift effect only as "
            "far as it goes beyond the same term without shift.",
            "Clustered is outside the family. It draws its internal split over row positions "
            "with its own seed fixed at 0. At one K the number of calibration points each "
            "position of the class layout keeps is therefore the same for every subset seed, "
            "while the rows in those positions change with the seed. Between K 10 and K 100 "
            "the mask, the share set aside for clustering and a class's position all change, "
            "so its same-class term compares calibration sets that are not the same rows, "
            "and at K 10 its rule skips clustering.",
            "Weighted fits its domain classifier on the K kept logits, so the classifier "
            "changes with K together with the label space, and its mean set size over K "
            "changes with it. Its same-class term mixes the two.",
            "At K 100 every seed holds all 100 classes, so the seeds vary only the calibration "
            "subsample and the pool and evaluation split.",
        ]
    return notes


def _opt3(value: float | None) -> str:
    return "none" if value is None else _signed(value)


def _term_text(stat: dict) -> str:
    text = f"{_signed(stat['mean'])} {_span(stat['ci_lo'], stat['ci_hi'])}"
    if stat["bonferroni_lo"] is not None:
        text += f" {{{_f3(stat['bonferroni_lo'])}, {_f3(stat['bonferroni_hi'])}}}"
    return text


CACHE_BY_CACHE_MAX = 5


def _across_conditions_lines(rows: Sequence[dict], present: Sequence[str],
                             caches: Sequence[str]) -> list[str]:
    first = _find(rows, method=present[0])["same_class"]
    if "n_families_negative" in first:
        lines = ["Across the corruption families, as description only: the families whose mean "
                 f"term is negative, of {len(CORRUPTIONS)}, then the smallest and the largest "
                 "family mean.", ""]
        table = []
        for method in present:
            row = _find(rows, method=method)
            cells = [f"{row[t]['n_families_negative']} of {len(CORRUPTIONS)}, "
                     f"{_signed(row[t]['family_min'])} to {_signed(row[t]['family_max'])}"
                     for t in FAMILY_TERMS]
            table.append([_METHOD_NAMES[method], *cells])
        return lines + _table(["method", "same-class term", "selection term"], table)
    if len(caches) > CACHE_BY_CACHE_MAX or "caches_means" not in first:
        return []
    lines = ["The two terms cache by cache, as description only: the mean of the same-class "
             "term, then of the selection term. No interval is given for a single cache.", ""]
    table = []
    for method in present:
        row = _find(rows, method=method)
        table.append([_METHOD_NAMES[method],
                      *(f"{_signed(same)} and {_signed(selection)}"
                        for same, selection in zip(row["same_class"]["caches_means"],
                                                   row["selection"]["caches_means"]))])
    return lines + _table(["method", *caches], table)


def _decomposition_lines(rows: Sequence[dict], present: Sequence[str],
                         caches: Sequence[str] = ()) -> list[str]:
    k_small, k_large = PRIMARY_PAIR
    lines = [f"Decomposition of the change from K {k_small} to K {k_large}. Each term is given "
             "with its 95% interval and, inside the family, its Bonferroni interval in braces. "
             "The last columns give the caches whose mean term is negative and the marks.", ""]
    table = []
    for method in present:
        row = _find(rows, method=method)
        negative = " and ".join(
            f"{row[t]['n_caches_negative']} of {row[t]['n_caches']}" for t in FAMILY_TERMS)
        marks = [] if row["in_family"] else ["outside the family"]
        marks += [f"{label} term bound" for term, label in (("same_class", "same-class"),
                                                            ("selection", "selection"))
                  if row[term]["marked"]]
        if row["selection"]["rb_reproduces"] is False:
            marks.append("selection exclusion not reproduced")
        table.append([_METHOD_NAMES[method], _term_text(row["same_class"]),
                      _term_text(row["selection"]), _signed(row["total"]), negative,
                      ", ".join(marks)])
    lines += _table(["method", "same-class term", "selection term", "sum", "caches negative",
                     "note"], table)

    lines += ["Limits of the coverage range. Shares of cells at 0 and at 1 for the worst class "
              f"at K {k_small} (a), the minimum over the K {k_small} classes inside the K "
              f"{k_large} problem (b) and the worst class at K {k_large} (c). Then, for each "
              "term, the share of bound cells and the mean over the other cells with their "
              "number; for the selection term also the part of its mean that the pinned cells "
              "contribute.", ""]
    table = []
    for method in present:
        row = _find(rows, method=method)
        limits = row["limits"]
        same, selection = row["same_class"], row["selection"]
        table.append([
            _METHOD_NAMES[method],
            *[f"{_f2(limits[name]['at_zero'])} / {_f2(limits[name]['at_one'])}"
              for name in ("worst_small", "kept_min_large", "worst_large")],
            f"{_f2(same['bound_share'])}, {_opt3(same['free_mean'])} ({same['n_free']})",
            f"{_f2(selection['bound_share'])}, {_opt3(selection['free_mean'])} "
            f"({selection['n_free']}), {_signed(selection['pinned_part'])}",
        ])
    lines += _table(["method", "a at 0 / at 1", "b at 0 / at 1", "c at 0 / at 1",
                     "same-class: bound share, other mean (cells)",
                     "selection: pinned share, other mean (cells), pinned part"], table)

    lines += ["Checks on the intervals. Skewness of the per-seed means behind each term, and "
              "the selection term with b replaced by its expectation over a random K "
              f"{k_small} subset of the K {k_large} classes.", ""]
    table = []
    for method in present:
        row = _find(rows, method=method)
        verdict = {None: "none to reproduce", True: "reproduced", False: "not reproduced"}[
            row["selection"]["rb_reproduces"]]
        table.append([_METHOD_NAMES[method],
                      f"{_opt3(row['same_class']['skewness'])} and "
                      f"{_opt3(row['selection']['skewness'])}",
                      _term_text(row["selection_rb"]), _opt3(row["selection_rb"]["skewness"]),
                      verdict if row["in_family"] else "outside the family"])
    lines += _table(["method", "skewness of the two terms", "selection term, expectation version",
                     "its skewness", "family exclusion"], table)

    lines += _across_conditions_lines(rows, present, caches)

    if "no_shift" in _find(rows, method=present[0])["same_class"]:
        lines += ["The same terms without shift (the views of the no-shift yardstick of this "
                  "backbone), and the per-seed difference of the term under shift from it, "
                  "each with its unadjusted 95% interval.", ""]
        table = []
        for method in present:
            row = _find(rows, method=method)
            cells = []
            for term in FAMILY_TERMS:
                for part in ("no_shift", "minus_no_shift"):
                    stat = row[term][part]
                    cells.append(f"{_signed(stat['mean'])} {_span(stat['ci_lo'], stat['ci_hi'])}")
            table.append([_METHOD_NAMES[method], *cells])
        lines += _table(["method", "same-class, no shift", "same-class, shift minus no shift",
                         "selection, no shift", "selection, shift minus no shift"], table)
    return lines


def _scope_lines(name: str, block: dict, design: str) -> list[str]:
    per_k = block["per_k"]
    present = [m for m in _METHOD_ORDER if any(r["method"] == m for r in per_k)]
    selected = " (selected on outcomes, descriptive)" if block["selected_on_outcomes"] else ""
    lines = [f"## {name}: {block['n_caches']} caches{selected}", "",
             "Mean worst-class coverage with its 95% t interval, unadjusted and conditional on "
             "the image set.", ""]
    reference_row = ["reference"] + [_f3(_find(per_k, method=present[0], k=k)["reference"])
                                     for k in K_VALUES]
    rows = [reference_row]
    for method in present:
        cells = [_find(per_k, method=method, k=k) for k in K_VALUES]
        rows.append([_METHOD_NAMES[method]]
                    + [f"{_f3(c['mean'])} {_span(c['ci_lo'], c['ci_hi'])}" for c in cells])
    lines += _table(_k_header("method"), rows)

    lines += ["Difference from the reference, with the share of cells at zero in brackets.", ""]
    rows = []
    for method in present:
        cells = [_find(per_k, method=method, k=k) for k in K_VALUES]
        rows.append([_METHOD_NAMES[method]]
                    + [f"{_signed(c['diff'])} ({_f2(c['zero_share'])})" for c in cells])
    lines += _table(_k_header("method"), rows)

    lines += ["Mean prediction-set size over K.", ""]
    rows = [[_METHOD_NAMES[m]] + [_f2(_find(per_k, method=m, k=k)["set_over_k"]) for k in K_VALUES]
            for m in present]
    lines += _table(_k_header("method"), rows)

    if "band_seed_mean" in per_k[0]:
        lines += [f"Caches whose seed mean is below / within / above the band of {Z_BOUND} "
                  "reference standard errors of a seed mean.", ""]
        rows = []
        for method in present:
            bands = [_find(per_k, method=method, k=k)["band_seed_mean"] for k in K_VALUES]
            rows.append([_METHOD_NAMES[method]]
                        + [f"{b['below']} / {b['within']} / {b['above']}" for b in bands])
        lines += _table(_k_header("method"), rows)

    if block["decomposition"]:
        lines += _decomposition_lines(block["decomposition"], present, block["caches"])
    return lines


def render_markdown(result: dict) -> str:
    config = result["config"]
    design = config["design"]
    lines = [
        f"# Class-count trend against the matched reference ({design} design)",
        "",
        f"Generated by `scripts/class_count_trend.py` from {config['n_files']} sweep files with "
        f"{config['n_seeds']} subset seeds each. Intervals are two-sided 95% t intervals on the "
        "per-seed means of a scope.",
        "",
        "## Matched reference",
        "",
        "Expected worst-class coverage of an exactly valid per-class method with "
        f"{config['n_cal_per_class']} calibration and {config['n_eval_per_class']} evaluation "
        "points per class.",
        "",
    ]
    lines += _table(["K", "reference mean", "single-draw sd"],
                    [[str(k), _f3(result["reference"][str(k)]["mean"]),
                      _f3(result["reference"][str(k)]["sd"])] for k in K_VALUES])
    if config["family"]:
        family = config["family"]
        lines += [f"The family holds {family['size']} intervals: the same-class and the "
                  f"selection term of {len(family['methods'])} methods on {len(family['scopes'])} "
                  f"backbone scopes. The Bonferroni level is {family['level']:.4f}.", ""]

    for name, block in result["scopes"].items():
        lines += _scope_lines(name, block, design)

    lines += ["## The moderate five caches one by one", "",
              "Mean worst-class coverage per cache. No interval is given for a single cache.", ""]
    for entry in result["moderate_caches"]:
        present = [m for m in _METHOD_ORDER if any(r["method"] == m for r in entry["per_k"])]
        rows = [[_METHOD_NAMES[m]] + [_f3(_find(entry["per_k"], method=m, k=k)["mean"])
                                      for k in K_VALUES] for m in present]
        lines += [entry["cache"], ""] + _table(_k_header("method"), rows)

    if result["no_shift"]:
        lines += ["## No-shift yardstick", "",
                  "Source rows stand in for the target, so no shift is present. The table gives "
                  "the mean worst-class coverage and, in brackets, its difference from the "
                  "reference.", ""]
        for backbone, block in result["no_shift"].items():
            present = [m for m in _METHOD_ORDER if any(r["method"] == m for r in block["per_k"])]
            rows = []
            for method in present:
                cells = [_find(block["per_k"], method=method, k=k) for k in K_VALUES]
                rows.append([_METHOD_NAMES[method]]
                            + [f"{_f3(c['mean'])} ({_signed(c['diff'])})" for c in cells])
            lines += [f"{backbone}, {block['n_views']} views of {block['base_cache']}", ""]
            lines += _table(_k_header("method"), rows)
            k_small, k_large = PRIMARY_PAIR
            lines += [f"{backbone}, the two terms of the change from K {k_small} to K {k_large} "
                      "without shift, each with its unadjusted 95% interval, then the share of "
                      "bound cells of each term.", ""]
            rows = []
            for method in present:
                row = _find(block["decomposition"], method=method)
                rows.append([_METHOD_NAMES[method], _term_text(row["same_class"]),
                             _term_text(row["selection"]),
                             f"{_f2(row['same_class']['bound_share'])} and "
                             f"{_f2(row['selection']['bound_share'])}"])
            lines += _table(["method", "same-class term", "selection term", "bound shares"], rows)

    lines += ["## How to read these numbers", ""]
    lines += [f"- {note}" for note in _reading_notes(result)]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="class_count_trend",
        description="Compare K-sweep files with the matched finite-sample reference.")
    parser.add_argument("--design", choices=("independent", "nested"), required=True)
    parser.add_argument("--inputs-glob", required=True,
                        help="Glob of the sweep files, one per cache.")
    parser.add_argument("--no-shift-glob", default=None,
                        help="Glob of the no-shift sweep files (nested design).")
    parser.add_argument("--expect-seeds", type=int, default=None,
                        help="Refuse a file whose subset_seeds differs from this.")
    parser.add_argument("--expect-no-shift", type=int, default=None,
                        help="Refuse the set unless exactly this many no-shift files verify.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="Monte Carlo draws of the reference.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check the files, print the verdict and write nothing.")
    args = parser.parse_args(argv)
    if not args.verify_only and (args.out_json is None or args.out_md is None):
        parser.error("--out-json and --out-md are required unless --verify-only is given")
    return args


def _collect(args: argparse.Namespace) -> tuple[list[Sweep], list[Sweep], list[str]]:
    paths = sorted(Path(p) for p in glob.glob(args.inputs_glob))
    if not paths:
        return [], [], [f"no file matches {args.inputs_glob}"]
    sweeps, problems = load_sweeps(paths, design=args.design, view="shift",
                                   n_seeds=args.expect_seeds)
    problems += grid_problems(sweeps)
    views: list[Sweep] = []
    if args.no_shift_glob:
        view_paths = sorted(Path(p) for p in glob.glob(args.no_shift_glob))
        views, view_problems = load_sweeps(view_paths, design=args.design, view="no_shift",
                                           n_seeds=args.expect_seeds)
        problems += view_problems
        if sweeps and any(view.n_seeds != sweeps[0].n_seeds for view in views):
            problems.append("a no-shift file differs from the sweep files in subset_seeds")
    if args.expect_no_shift is not None and len(views) != args.expect_no_shift:
        problems.append(f"{len(views)} no-shift file(s) verify, expected {args.expect_no_shift}")
    problems += set_problems(sweeps, views, args.design)
    return sweeps, views, problems


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sweeps, views, problems = _collect(args)
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        print(f"VERIFY FAILED: {len(problems)} problem(s)")
        return 1
    versions = sorted({s.provenance.get("code_version") for s in sweeps + views
                       if s.provenance.get("code_version")})
    print(f"VERIFY OK: {len(sweeps)} sweep files, {len(views)} no-shift files, "
          f"{sweeps[0].n_seeds} subset seeds, {args.design} design, "
          f"code version {versions[0] if versions else 'not recorded'}")
    for backbone in BACKBONES:
        digests = {s.provenance.get("source_cal_sha256") for s in sweeps
                   if s.key.backbone == backbone and s.provenance.get("source_cal_sha256")}
        if digests:
            print(f"{backbone}: {len(digests)} distinct source set(s) among its caches")
    if args.verify_only:
        return 0

    result = analyse(sweeps, design=args.design, no_shift=views or None, reps=args.reps)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
