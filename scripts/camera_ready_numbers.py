"""Numbers and table bodies of the camera-ready paper, written from committed artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import number_registry as registry_module  # noqa: E402
from class_count_trend import FAMILY_TERMS, READINGS, seed_interval, term_reading  # noqa: E402

from pcpbench.metrics.validity_floor import exact_min_coverage  # noqa: E402

PAPER_DIR = "paper/camera_ready"
GENERATED_DIR = f"{PAPER_DIR}/generated"
ALL_CACHE_TREND = "outputs/analysis/all_cache_trend.json"
NESTED_TREND = "outputs/analysis/nested_trend.json"
VALIDITY_FLOOR = registry_module.VALIDITY_FLOOR
LABEL_SWEEP = registry_module.LABEL_SWEEP
NO_SHIFT_DIR = registry_module.NO_SHIFT_DIR
ANALYSIS_DIR = registry_module.ANALYSIS_DIR
Z_BOUND = registry_module.Z_BOUND
REVIEWED_VERSION = registry_module.REVIEWED_VERSION

KEY_RE = re.compile(r"[A-Za-z0-9._-]+")
USE_RE = re.compile(r"\\crn\{([^{}]*)\}")

K_VALUES = (10, 25, 50, 100)
SCOPE_SLUG = {"ViT-B/32": "vitb", "ViT-L/14": "vitl"}
MODERATE_SCOPE = "moderate five (ViT-B/32)"
ALL_SCOPE_SLUG = {**SCOPE_SLUG, MODERATE_SCOPE: "moderate"}
HEADLINE_ORDER = ("oracle", "mondrian", "split", "weighted", "ppi", "acp")
SUPPLEMENTARY = "clustered"
METHOD_NAME = {"oracle": "Oracle", "mondrian": "Source-Mondrian", "split": "Split",
               "weighted": "Weighted", "ppi": "PPI", "acp": "ACP", "clustered": "Clustered"}
DELTA_ORDER = ("acp", "ppi", "oracle", "split", "mondrian", "weighted")
VITL_ORDER = ("acp", "ppi", "split", "oracle", "mondrian", "weighted")
IMAGENET_ORDER = ("split", "mondrian", "ppi", "oracle", "weighted", "acp")
LABEL_BUDGETS = (0, 10, 25, 50)
NTU_ZERO_CLASS_METHODS = ("split", "mondrian", "weighted")
CI_LEVEL = 0.95
NO_SHIFT_METHODS = ("mondrian", "oracle", "split")
PICK_METHODS = ("split", "mondrian")
WEIGHTED = "weighted"
ACP = "acp"
REPLICATION_METHODS = ("oracle", "mondrian", "split", "weighted", "ppi", "acp")
DERIVED_NOTES = {
    "derived.guarantee.n50": "ceil(0.9 * 51) / 51, the expected coverage of split conformal "
                             "with 50 calibration points at level 0.90",
    "derived.crossdataset.reference_decline": "validity_floor rows cifar10c minus cifar100c, "
                                              "floor_mean, from the unrounded values",
    "derived.trend.vitb.oracle.max_abs_diff": "max over K of |trend.vitb.oracle.K.diff|",
    "derived.trend.vitl.oracle.max_abs_diff": "max over K of |trend.vitl.oracle.K.diff|",
    "derived.imagenet.<method>.<column>": "mean over the method's ten ImageNet-1k rows of the "
                                          "rerun board, per printed column of tab:imagenet",
    "derived.imagenet.clustered.max_cal_points_per_class": "max over the ImageNet-1k clustered "
                                                           "rows and seeds of "
                                                           "n_cal_per_class_per_seed",
    "derived.imagenet.split.cal_points_per_class": "the single value of n_cal_per_class_per_seed "
                                                   "on the ImageNet-1k split rows",
    "derived.delta.<method>.*": "the cells of tab:delta on the rerun board (n_info, "
                                "informative, all_cell), with .magnitude the absolute value; "
                                "for acp, info_min_set_over_k is the smallest set_over_k among "
                                "the CIFAR-10-C and CIFAR-100-C rows of its informative pairs",
    "derived.recovered.<method>.*": "over the headline rows above the divider that label_regime "
                                    "calls recovered: n_rows per method, and for acp the "
                                    "smallest and largest set_over_k of its rows",
    "derived.vitl.<method>.*": "the cells of tab:vitl on the rerun board (vitl, vitb), with "
                               ".magnitude the absolute value; n_pairs and n_info count the "
                               "matched and the informative ViT-L/14 pairs behind the vitl cell; "
                               "for acp, info_min_set_over_k is the smallest set_over_k among "
                               "the rows of its informative ViT-L/14 pairs",
    "derived.weighted_groups.<backbone>.*": "weighted's informative pairs of tab:delta (vitb) "
                                            "and tab:vitl (vitl) in two groups: n_full, the pairs "
                                            "whose CIFAR-100-C row has set_over_k of at least the "
                                            "saturation filter's 0.90, with n_full_demoted of them "
                                            "demoted by the guard and full_min_worst their lowest "
                                            "CIFAR-100-C worst_class; n_rest and rest_mean, the "
                                            "other pairs and their mean change, .magnitude its "
                                            "absolute value",
    "derived.no_shift.*": "over the distinct controls of outputs/noshift/ (two controls are one "
                          "when every method's summary agrees): n_distinct; per method "
                          "n_within_two_sd and n_below_two_sd read the summary's z against 2, "
                          "n_within_two_se and n_below_two_se read z * sqrt(n_seeds) against 2",
    "derived.selection.*": "over the ViT-B/32 controlled-sweep files: n_vitb_caches, "
                           "n_qualifying (five-seed mean worst_class above zero at K 10 for split "
                           "and for mondrian) and n_picked, the five moderate caches",
    "derived.reference.closed_form_gap_bound": "max over the rows of validity_floor.json of "
                                               "|floor_mean - exact_min_coverage(K, n_cal, "
                                               "n_eval).mean|, rounded up to four decimals",
    "derived.imagenet.<method>.*": "over the method's ImageNet-1k rows of the rerun board: "
                                   "n_cells, n_zero_worst (worst_class == 0), n_demoted, "
                                   "n_zero_worst_kept (zero and not demoted), and min/max of "
                                   "set_over_k",
    "derived.monotonicity.*": "exhibit_monotonicity of scripts/analyze_leaderboard.py on the "
                              "six-method headline of the rerun board; tolerance is its "
                              "MONOTONICITY_TOLERANCE",
    "derived.oracle.cifar100c.*": "over the oracle's headline rows on CIFAR-100-C: n_rows, and "
                                  "n_within_two_se, the rows whose unrounded worst_class lies "
                                  "within 2 * floor_sd / sqrt(seeds) of floor_mean "
                                  "(validity_floor row cifar100c, seeds from the row JSON)",
    "derived.label.<method>.m<m>.marginal": "marginal_mean of the label-budget summary row",
    "derived.label.<method>.m<m>.ci_lo/ci_hi": "t interval on the per-seed worst-class values "
                                               "(10 seeds), clipped to [0, 1]",
    "derived.label.ppi_minus_oracle.m<m>.*": "t interval on the per-seed differences, PPI "
                                             "minus oracle on the same seed",
    "derived.nested.family.*": "over the family of nested_trend.json (its methods, the two "
                               "terms, the two backbone scopes), the reading of each term by "
                               "class_count_trend.term_reading: n_bound, n_contains, n_withheld, "
                               "n_negative, n_positive, and the same per method under "
                               "<method>.n_<reading>; n_negative_beyond_no_shift counts the "
                               "negative readings whose paired interval for the term under shift "
                               "minus the term without shift lies below zero, "
                               "n_negative_short_of_no_shift those whose interval lies above it; "
                               "negative.min_bound_share and negative.max_bound_share are the "
                               "smallest and largest share of bound cells among the negative "
                               "readings",
    "derived.nested.replication.*": "the five moderate caches, nested run against independent "
                                    "run, over oracle, mondrian, split, weighted, ppi and acp: "
                                    "K100.max_abs_diff is the largest |difference of the two "
                                    "means| at K 100, rounded up to three decimals; "
                                    "K10.n_inside counts the methods whose nested K 10 mean "
                                    "lies inside the independent run's five-seed interval, "
                                    "n_cells_inside the same over every K (n_cells method by K "
                                    "cells)",
    "derived.ntu2nucla.labelfree.worst_class": "worst_class of the ntu2nucla rows of split, "
                                               "mondrian and weighted on the rerun board; the "
                                               "run stops when the three printed values differ",
}


def f3(x: float) -> str:
    return f"{x:.3f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def f1(x: float) -> str:
    return f"{x:.1f}"


def ceil3(x: float) -> str:
    """Three decimals, rounded up, for a largest difference the prose states as 'within'."""
    return f"{math.ceil(x * 1e3 - 1e-9) / 1e3:.3f}"


def ceil4(x: float) -> str:
    """Four decimals, rounded up: the form of a bound that the prose states as 'within'."""
    return f"{math.ceil(x * 1e4 - 1e-9) / 1e4:.4f}"


def signed3(x: float) -> str:
    """Three decimals with a sign; a value that rounds to zero prints unsigned."""
    text = f"{x:+.3f}"
    return "0.000" if text in ("+0.000", "-0.000") else text


def dot3(x: float) -> str:
    """Three decimals without the leading zero, the compact form of the wide tables."""
    text = f"{x:.3f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def sdot3(x: float) -> str:
    """Signed three decimals without the leading zero; a value that rounds to zero prints
    unsigned."""
    text = signed3(x)
    for lead in ("+0.", "-0.", "0."):
        if text.startswith(lead):
            return lead.replace("0.", ".") + text[len(lead):]
    return text


def dot2(x: float) -> str:
    text = f"{x:.2f}"
    return text[1:] if text.startswith("0.") else text


def _load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def registry_numbers(root: Path) -> dict[str, str]:
    """The registry's entries on the files as committed."""
    out: dict[str, str] = {}
    for entry in registry_module.build_registry(root):
        if KEY_RE.fullmatch(entry.name):
            if entry.name in out:
                raise ValueError(f"the registry prints {entry.name} twice")
            out[entry.name] = entry.value
    return out


def trend_numbers(result: dict, prefix: str) -> dict[str, str]:
    """Keys of one ``class_count_trend.py`` result under ``prefix``."""
    config = result["config"]
    out = {f"{prefix}.n_files": str(config["n_files"]),
           f"{prefix}.n_seeds": str(config["n_seeds"])}
    if config.get("family"):
        out[f"{prefix}.family.size"] = str(config["family"]["size"])
    for k, ref in result["reference"].items():
        out[f"{prefix}.reference.K{k}.mean"] = f3(ref["mean"])
        out[f"{prefix}.reference.K{k}.sd"] = f3(ref["sd"])
    for scope, block in result["scopes"].items():
        slug = ALL_SCOPE_SLUG[scope]
        out[f"{prefix}.{slug}.n_caches"] = str(block["n_caches"])
        for row in block["per_k"]:
            base = f"{prefix}.{slug}.{row['method']}.K{row['k']}"
            out.update(_per_k_keys(base, row))
            for band, tag in (("band_single_draw", "single_draw"), ("band_seed_mean", "seed_mean")):
                for name in ("below", "within", "above"):
                    if band in row:
                        out[f"{base}.{tag}.n_{name}"] = str(row[band][name])
        for row in block["decomposition"] or []:
            base = f"{prefix}.{slug}.{row['method']}.decomp"
            out[f"{base}.total"] = signed3(row["total"])
            out[f"{base}.mean_class_change"] = signed3(row["mean_class_change"])
            out[f"{base}.zero_share_k_small"] = f2(row["zero_share_k_small"])
            out[f"{base}.zero_share_k_large"] = f2(row["zero_share_k_large"])
            for term in FAMILY_TERMS:
                stat = row[term]
                out[f"{base}.{term}.bound_share"] = f2(stat["bound_share"])
                if stat["marked"]:
                    continue
                out[f"{base}.{term}.n_caches_negative"] = str(stat["n_caches_negative"])
                if "n_families_negative" in stat:
                    out[f"{base}.{term}.n_families_negative"] = str(stat["n_families_negative"])
                if slug == ALL_SCOPE_SLUG[MODERATE_SCOPE]:
                    for cache, value in zip(block["caches"], stat["caches_means"], strict=True):
                        out[f"{prefix}.cache.{cache}.{row['method']}.decomp.{term}.mean"] = \
                            signed3(value)
                for field in ("mean", "ci_lo", "ci_hi", "bonferroni_lo", "bonferroni_hi",
                              "free_mean"):
                    if stat[field] is not None:
                        out[f"{base}.{term}.{field}"] = signed3(stat[field])
                for name in ("no_shift", "minus_no_shift"):
                    for field, value in (stat.get(name) or {}).items():
                        out[f"{base}.{term}.{name}.{field}"] = signed3(value)
    for cache in result.get("moderate_caches") or []:
        for row in cache["per_k"]:
            base = f"{prefix}.cache.{cache['cache']}.{row['method']}.K{row['k']}"
            out[f"{base}.mean"] = f3(row["mean"])
            out[f"{base}.zero_share"] = f2(row["zero_share"])
    for backbone, block in (result.get("no_shift") or {}).items():
        slug = SCOPE_SLUG[backbone]
        out[f"{prefix}.noshift.{slug}.n_views"] = str(block["n_views"])
        for row in block["per_k"]:
            out.update(_per_k_keys(f"{prefix}.noshift.{slug}.{row['method']}.K{row['k']}", row))
    return out


def nested_family_numbers(nested: dict) -> dict[str, str]:
    """Counts of the readings over the pre-specified family of the nested run."""
    family = nested["config"]["family"]
    readings: list[tuple[str, str, dict]] = []
    for scope in family["scopes"]:
        rows = {r["method"]: r for r in nested["scopes"][scope]["decomposition"]}
        for method in family["methods"]:
            for term in family["terms"]:
                readings.append((method, term_reading(rows[method], term), rows[method][term]))
    if len(readings) != family["size"]:
        raise ValueError(f"the family holds {family['size']} intervals, found {len(readings)}")
    if any(reading == "outside" for _, reading, _ in readings):
        raise ValueError("a row of the family is marked as outside it")
    out: dict[str, str] = {}
    counted = [r for r in READINGS if r != "outside"]
    for reading in counted:
        out[f"derived.nested.family.n_{reading}"] = str(sum(r == reading for _, r, _ in readings))
        for method in family["methods"]:
            out[f"derived.nested.family.{method}.n_{reading}"] = str(
                sum(m == method and r == reading for m, r, _ in readings))
    negative = [stat for _, reading, stat in readings if reading == "negative"]
    paired = [stat["minus_no_shift"] for stat in negative]
    out["derived.nested.family.n_negative_beyond_no_shift"] = str(sum(d["ci_hi"] < 0 for d in paired))
    out["derived.nested.family.n_negative_short_of_no_shift"] = str(sum(d["ci_lo"] > 0 for d in paired))
    if negative:
        shares = [stat["bound_share"] for stat in negative]
        out["derived.nested.family.negative.min_bound_share"] = f2(min(shares))
        out["derived.nested.family.negative.max_bound_share"] = f2(max(shares))
    return out


def reviewed_numbers(root: Path) -> dict[str, str]:
    """The values the reviewed version printed, as ``REVIEWED_VERSION`` holds them."""
    values = _load(root, REVIEWED_VERSION)["values"]
    stray = sorted(k for k in values if not k.startswith("reviewed."))
    if stray:
        raise ValueError(f"{REVIEWED_VERSION} holds keys outside reviewed.*: {stray}")
    return dict(values)


def nested_replication_numbers(trend: dict, nested: dict) -> dict[str, str]:
    """The five moderate caches in the nested run beside the independent run."""
    def index(result: dict) -> dict[tuple[str, int], dict]:
        return {(r["method"], r["k"]): r for r in result["scopes"][MODERATE_SCOPE]["per_k"]}

    old, new = index(trend), index(nested)
    k_small, k_large = K_VALUES[0], K_VALUES[-1]
    gaps = [abs(new[(m, k_large)]["mean"] - old[(m, k_large)]["mean"]) for m in REPLICATION_METHODS]
    inside = {(m, k): old[(m, k)]["ci_lo"] <= new[(m, k)]["mean"] <= old[(m, k)]["ci_hi"]
              for m in REPLICATION_METHODS for k in K_VALUES}
    return {f"derived.nested.replication.K{k_large}.max_abs_diff": ceil3(max(gaps)),
            f"derived.nested.replication.K{k_small}.n_inside": str(
                sum(inside[(m, k_small)] for m in REPLICATION_METHODS)),
            "derived.nested.replication.n_cells_inside": str(sum(inside.values())),
            "derived.nested.replication.n_cells": str(len(inside)),
            "derived.nested.replication.n_methods": str(len(REPLICATION_METHODS))}


def _per_k_keys(base: str, row: dict) -> dict[str, str]:
    return {f"{base}.mean": f3(row["mean"]), f"{base}.ci_lo": f3(row["ci_lo"]),
            f"{base}.ci_hi": f3(row["ci_hi"]), f"{base}.diff": signed3(row["diff"]),
            f"{base}.zero_share": f2(row["zero_share"]),
            f"{base}.set_over_k": f2(row["set_over_k"])}


def _imagenet_columns(rows: list, method: str) -> list[str]:
    cells = [r for r in rows if r.dataset.startswith("imagenet1k") and r.method == method]
    if len(cells) != 10:
        raise ValueError(f"{method} has {len(cells)} ImageNet-1k rows, the appendix states 10")
    return [f3(mean(r.marginal for r in cells)), f3(mean(r.worst_class for r in cells)),
            f3(mean(r.frac_undercovered for r in cells)), f1(mean(r.mean_set_size for r in cells)),
            f3(mean(r.set_over_k for r in cells))]


def derived_numbers(root: Path, trend: dict, board: list) -> dict[str, str]:
    floor = {(r["setting"], r["K"]): r for r in _load(root, VALIDITY_FLOOR)["rows"]
             if r["setting"] in ("cifar10c", "cifar100c")}
    out = {
        "derived.guarantee.n50": f3(math.ceil(0.9 * 51) / 51),
        "derived.crossdataset.reference_decline": f3(floor[("cifar10c", 10)]["floor_mean"]
                                                      - floor[("cifar100c", 100)]["floor_mean"]),
    }
    for scope, slug in SCOPE_SLUG.items():
        diffs = [abs(r["diff"]) for r in trend["scopes"][scope]["per_k"] if r["method"] == "oracle"]
        out[f"derived.trend.{slug}.oracle.max_abs_diff"] = f3(max(diffs))
    names = ("marginal", "worst_class", "frac_undercovered", "mean_set_size", "set_over_k")
    for method in (*HEADLINE_ORDER, SUPPLEMENTARY):
        for name, cell in zip(names, _imagenet_columns(board, method)):
            out[f"derived.imagenet.{method}.{name}"] = cell
    out.update(_imagenet_clustered_calibration(root))
    out.update(board_numbers(root, board))
    out.update(label_numbers(root))
    out.update(no_shift_distinct(root))
    out.update(selection_numbers(root, trend))
    out.update(reference_closed_form(root))
    return out


def no_shift_distinct(root: Path) -> dict[str, str]:
    """Counts over the no-shift controls that are distinct."""
    groups: dict[str, dict[str, dict]] = {}
    for path in sorted((root / NO_SHIFT_DIR).glob("*.json")):
        summary = {e["method"]: e for e in json.loads(path.read_text(encoding="utf-8"))["summary"]}
        signature = json.dumps([[m, e["worst_class_mean"], e["marginal_mean"], e["mean_set_size_mean"]]
                                for m, e in sorted(summary.items())])
        groups.setdefault(signature, summary)
    out = {"derived.no_shift.n_distinct": str(len(groups))}
    for method in NO_SHIFT_METHODS:
        entries = [summary[method] for summary in groups.values()]
        if any(e["floor_status"] != "ok" for e in entries):
            raise ValueError(f"{method} has a no-shift control without a reference")
        for tag, values in (("sd", [e["z"] for e in entries]),
                            ("se", [e["z"] * math.sqrt(e["n_seeds"]) for e in entries])):
            base = f"derived.no_shift.{method}"
            out[f"{base}.n_within_two_{tag}"] = str(sum(abs(v) <= Z_BOUND for v in values))
            out[f"{base}.n_below_two_{tag}"] = str(sum(v < -Z_BOUND for v in values))
    return out


def selection_numbers(root: Path, trend: dict) -> dict[str, str]:
    """How many ViT-B/32 caches meet the condition under which the five moderate caches were
    picked by hand: split and Source-Mondrian both keep a five-seed mean worst-class coverage
    above zero at the smallest K."""
    k = min(K_VALUES)
    caches = trend["scopes"]["ViT-B/32"]["caches"]
    qualifying = []
    for cache in caches:
        records = _load(root, f"{ANALYSIS_DIR}/controlled_k_sweep_{cache}.json")["records"]
        means = [mean(r["worst_class"] for r in records if r["method"] == m and r["k"] == k)
                 for m in PICK_METHODS]
        if min(means) > 0:
            qualifying.append(cache)
    picked = trend["scopes"][MODERATE_SCOPE]["caches"]
    if not set(picked) <= set(qualifying):
        raise ValueError("a moderate cache does not meet the condition the prose states")
    return {"derived.selection.n_vitb_caches": str(len(caches)),
            "derived.selection.n_qualifying": str(len(qualifying)),
            "derived.selection.n_picked": str(len(picked))}


def reference_closed_form(root: Path) -> dict[str, str]:
    """A bound on the distance between a printed Monte Carlo reference and its closed form."""
    payload = _load(root, VALIDITY_FLOOR)
    gaps = [abs(row["floor_mean"] - exact_min_coverage(row["K"], np.asarray(row["n_cal"]),
                                                      np.asarray(row["n_eval"]),
                                                      payload["alpha"]).mean)
            for row in payload["rows"]]
    return {"derived.reference.closed_form_gap_bound": ceil4(max(gaps))}


def _with_magnitude(key: str, cell: str) -> dict[str, str]:
    return {key: cell, f"{key}.magnitude": cell.lstrip("+-")}


def board_numbers(root: Path, board: list) -> dict[str, str]:
    """The table cells the cross-dataset prose quotes, and the monotonicity counts."""
    al = registry_module.analyze_module(root)
    tables = registry_module._table_values(board, al)
    out: dict[str, str] = {}
    for method, (n_info, informative, all_cell) in tables["tab:delta"].items():
        out[f"derived.delta.{method}.n_info"] = n_info
        out.update(_with_magnitude(f"derived.delta.{method}.informative", informative))
        out.update(_with_magnitude(f"derived.delta.{method}.all_cell", all_cell))
    for method, (vitl, vitb) in tables["tab:vitl"].items():
        out.update(_with_magnitude(f"derived.vitl.{method}.vitl", vitl))
        out.update(_with_magnitude(f"derived.vitl.{method}.vitb", vitb))
    out.update(_vitl_counts(board, al, tables["tab:vitl"]))
    out.update(_acp_pair_set_shares(board, al, tables))
    out.update(_recovered_by_method(board, al))
    out.update(weighted_groups(board, al, tables))
    out.update(_imagenet_counts(board))
    out.update(_ntu_zero_class(board))
    out.update(_oracle_on_reference(root, board, al))
    violations, dropped = al.exhibit_monotonicity(al.select_headline(board))
    out["derived.monotonicity.tolerance"] = f2(al.MONOTONICITY_TOLERANCE)
    out["derived.monotonicity.n_violations"] = str(len(violations))
    out["derived.monotonicity.n_dropped_saturated"] = str(dropped)
    by_method = Counter(v.method for v in violations)
    for method in HEADLINE_ORDER:
        out[f"derived.monotonicity.{method}"] = str(by_method.get(method, 0))
    return out


def _oracle_on_reference(root: Path, board: list, al) -> dict[str, str]:
    floor = next(r for r in _load(root, VALIDITY_FLOOR)["rows"]
                 if (r["setting"], r["K"]) == ("cifar100c", 100))
    rows = [r for r in al.select_headline(board)
            if r.method == "oracle" and al.parse_dataset_name(r.dataset).family == "cifar100c"]
    within = 0
    for row in rows:
        exact = _load(root, f"{registry_module.ROWS_DIR}/{row.dataset}_oracle.json")
        sizes = {(n, m)
                 for cal, ev in zip(exact["n_cal_per_class_per_seed"], exact["n_eval_per_class_per_seed"])
                 for n, m in zip(cal, ev)}
        if exact["n_classes"] != floor["K"] or sizes != {(floor["n_cal"], floor["n_eval"])}:
            raise ValueError(f"{row.dataset} is not in the setting of the cifar100c reference row")
        half = 2.0 * floor["floor_sd"] / math.sqrt(exact["n_valid_seeds"])
        within += abs(exact["worst_class"] - floor["floor_mean"]) <= half
    return {"derived.oracle.cifar100c.n_rows": str(len(rows)),
            "derived.oracle.cifar100c.n_within_two_se": str(within)}


def _vitl_counts(board: list, al, vitl_table: dict) -> dict[str, str]:
    rows = [r for r in board if r.structured and r.backbone == "ViT-L"
            and r.method in al.HEADLINE_METHODS]
    cells, _ = al.exhibit_k_scaling(rows)
    out: dict[str, str] = {}
    for method, (printed, _) in vitl_table.items():
        mine = [c for c in cells if c.method == method]
        informative = [c.delta_worst for c in mine if c.informative]
        if f"{mean(informative):+.3f}" != printed:
            raise ValueError(f"ViT-L/14 {method}: the selection here does not give the printed cell")
        out[f"derived.vitl.{method}.n_pairs"] = str(len(mine))
        out[f"derived.vitl.{method}.n_info"] = str(len(informative))
    return out


def _acp_pair_set_shares(board: list, al, tables: dict) -> dict[str, str]:
    selections = {
        "derived.delta.acp": ([r for r in al.select_headline(board) if r.structured],
                              tables["tab:delta"][ACP][1], tables["tab:delta"][ACP][0]),
        "derived.vitl.acp": ([r for r in board if r.structured and r.backbone == "ViT-L"
                              and r.method in al.HEADLINE_METHODS], tables["tab:vitl"][ACP][0], None),
    }
    out: dict[str, str] = {}
    for base, (rows, printed_mean, printed_n) in selections.items():
        cells, _ = al.exhibit_k_scaling(rows)
        informative = [c for c in cells if c.method == ACP and c.informative]
        if (f"{mean(c.delta_worst for c in informative):+.3f}" != printed_mean
                or (printed_n is not None and str(len(informative)) != printed_n)):
            raise ValueError(f"{base}: the selection here does not give the printed cells")
        by_key: dict[tuple, object] = {}
        for row in rows:
            if row.method == ACP:
                key = (row.corruption, row.severity, row.backbone, row.k)
                if key in by_key:
                    raise ValueError(f"two ACP rows for {key}")
                by_key[key] = row
        shares = [by_key[(c.corruption, c.severity, c.backbone, k)].set_over_k
                  for c in informative for k in (c.low_k, c.high_k)]
        out[f"{base}.info_min_set_over_k"] = f2(min(shares))
    return out


def _recovered_by_method(board: list, al) -> dict[str, str]:
    recovered = [r for r in al.select_headline(board)
                 if not r.demoted and al.label_regime(r.marginal, r.worst_class) == al.RECOVERED]
    out = {f"derived.recovered.{method}.n_rows": str(sum(r.method == method for r in recovered))
           for method in HEADLINE_ORDER}
    acp = [r.set_over_k for r in recovered if r.method == ACP]
    if acp:
        out["derived.recovered.acp.min_set_over_k"] = f2(min(acp))
        out["derived.recovered.acp.max_set_over_k"] = f2(max(acp))
    return out


def weighted_groups(board: list, al, tables: dict) -> dict[str, str]:
    """Weighted's informative cross-dataset pairs in two groups, per backbone."""
    selections = {
        "vitb": ([r for r in al.select_headline(board) if r.structured],
                 tables["tab:delta"][WEIGHTED][1]),
        "vitl": ([r for r in board if r.structured and r.backbone == "ViT-L"
                  and r.method in al.HEADLINE_METHODS], tables["tab:vitl"][WEIGHTED][0]),
    }
    out: dict[str, str] = {}
    for slug, (rows, printed) in selections.items():
        cells, _ = al.exhibit_k_scaling(rows)
        informative = [c for c in cells if c.method == WEIGHTED and c.informative]
        if f"{mean(c.delta_worst for c in informative):+.3f}" != printed:
            raise ValueError(f"weighted on {slug}: the selection here does not give the printed cell")
        high: dict[tuple, object] = {}
        for row in rows:
            if row.method == WEIGHTED:
                key = (row.corruption, row.severity, row.k)
                if key in high:
                    raise ValueError(f"two weighted rows for {key} on {slug}")
                high[key] = row
        full = [c for c in informative
                if high[(c.corruption, c.severity, c.high_k)].set_over_k >= al.TRIVIAL_VACUOUS_SETK]
        rest = [c for c in informative if c not in full]
        if not full or not rest:
            raise ValueError(f"weighted on {slug}: one of the two groups is empty")
        base = f"derived.weighted_groups.{slug}"
        out[f"{base}.n_full"] = str(len(full))
        out[f"{base}.n_full_demoted"] = str(
            sum(bool(high[(c.corruption, c.severity, c.high_k)].demoted) for c in full))
        out[f"{base}.n_rest"] = str(len(rest))
        out[f"{base}.full_min_worst"] = f3(min(c.worst_class_high_k for c in full))
        out.update(_with_magnitude(f"{base}.rest_mean", f"{mean(c.delta_worst for c in rest):+.3f}"))
    return out


def _imagenet_counts(board: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for method in (*HEADLINE_ORDER, SUPPLEMENTARY):
        cells = [r for r in board if r.dataset.startswith("imagenet1k") and r.method == method]
        out[f"derived.imagenet.{method}.n_cells"] = str(len(cells))
        out[f"derived.imagenet.{method}.n_zero_worst"] = str(sum(r.worst_class == 0 for r in cells))
        out[f"derived.imagenet.{method}.n_demoted"] = str(sum(bool(r.demoted) for r in cells))
        out[f"derived.imagenet.{method}.n_zero_worst_kept"] = str(
            sum(r.worst_class == 0 and not r.demoted for r in cells))
        out[f"derived.imagenet.{method}.min_set_over_k"] = f2(min(r.set_over_k for r in cells))
        out[f"derived.imagenet.{method}.max_set_over_k"] = f2(max(r.set_over_k for r in cells))
    return out


def _ntu_zero_class(board: list) -> dict[str, str]:
    values = {r.method: f3(r.worst_class) for r in board
              if r.dataset == "ntu2nucla" and r.method in NTU_ZERO_CLASS_METHODS}
    if sorted(values) != sorted(NTU_ZERO_CLASS_METHODS):
        raise ValueError(f"ntu2nucla rows on the board: {sorted(values)}, "
                         f"expected {sorted(NTU_ZERO_CLASS_METHODS)}")
    if len(set(values.values())) != 1:
        raise ValueError(f"the ntu2nucla worst-class values differ: {values}")
    return {"derived.ntu2nucla.labelfree.worst_class": next(iter(values.values()))}


def _imagenet_clustered_calibration(root: Path) -> dict[str, str]:
    paths = sorted((root / "outputs").glob("imagenet1k_*_clustered.json"))
    if len(paths) != 10:
        raise ValueError(f"{len(paths)} ImageNet-1k clustered rows, the appendix states 10")
    most, split_sizes = 0, set()
    for path in paths:
        clustered = json.loads(path.read_text(encoding="utf-8"))["n_cal_per_class_per_seed"]
        split_path = path.with_name(path.name.replace("_clustered.json", "_split.json"))
        split = json.loads(split_path.read_text(encoding="utf-8"))["n_cal_per_class_per_seed"]
        most = max(most, max(max(seed) for seed in clustered))
        split_sizes |= {n for seed in split for n in seed}
    if len(split_sizes) != 1:
        raise ValueError(f"split calibrates on {sorted(split_sizes)} points per class on ImageNet-1k")
    return {"derived.imagenet.clustered.max_cal_points_per_class": str(most),
            "derived.imagenet.split.cal_points_per_class": str(split_sizes.pop())}


def _label_seed_values(root: Path) -> dict[tuple[str, int], list[float]]:
    payload = _load(root, LABEL_SWEEP)
    cells: dict[tuple[str, int], dict[int, float]] = {}
    for row in payload["per_seed"]:
        if row["worst_class"] is not None:
            cells.setdefault((row["method"], row["m"]), {})[row["seed"]] = float(row["worst_class"])
    return {key: [seeds[s] for s in sorted(seeds)] for key, seeds in cells.items()}


def label_numbers(root: Path) -> dict[str, str]:
    """Marginal coverage and t intervals of the label-budget sweep."""
    out: dict[str, str] = {}
    for row in _load(root, LABEL_SWEEP)["summary"]:
        if row.get("marginal_mean") is not None:
            out[f"derived.label.{row['method']}.m{row['m']}.marginal"] = f3(row["marginal_mean"])
    values = _label_seed_values(root)
    for (method, m), seeds in values.items():
        lo, hi = seed_interval(np.asarray(seeds), CI_LEVEL)
        out[f"derived.label.{method}.m{m}.ci_lo"] = f3(max(lo, 0.0))
        out[f"derived.label.{method}.m{m}.ci_hi"] = f3(min(hi, 1.0))
    for m in LABEL_BUDGETS[1:]:
        ppi, oracle = values[("ppi", m)], values[("oracle", m)]
        if len(ppi) != len(oracle):
            raise ValueError(f"PPI and the oracle have different seed counts at m={m}")
        diff = np.asarray(ppi) - np.asarray(oracle)
        lo, hi = seed_interval(diff, CI_LEVEL)
        base = f"derived.label.ppi_minus_oracle.m{m}"
        out[f"{base}.diff_mean"] = signed3(float(diff.mean()))
        out[f"{base}.diff_ci_lo"] = signed3(lo)
        out[f"{base}.diff_ci_hi"] = signed3(hi)
    return out


def build_pool(root: Path = _REPO_ROOT) -> dict[str, str]:
    root = Path(root)
    al = registry_module.analyze_module(root)
    board = registry_module.rerun_board(root, al)
    trend = _load(root, ALL_CACHE_TREND)
    nested = _load(root, NESTED_TREND)
    parts = [registry_numbers(root), trend_numbers(trend, "trend"),
             trend_numbers(nested, "nested"), derived_numbers(root, trend, board),
             nested_family_numbers(nested), nested_replication_numbers(trend, nested),
             reviewed_numbers(root)]
    pool: dict[str, str] = {}
    for part in parts:
        clash = sorted(set(part) & set(pool))
        if clash:
            raise ValueError(f"a key comes from two parts of the pool: {clash[:5]}")
        pool.update(part)
    return pool


def source_files(root: Path) -> list[Path]:
    paper = root / PAPER_DIR
    return [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]


COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def strip_comments(text: str) -> str:
    """The TeX source without its comments: from an unescaped % to the end of a line."""
    return COMMENT_RE.sub("", text)


NAMEDEF_RE = re.compile(r"\\@namedef\{crn@([^{}]*)\}")


def used_keys(root: Path) -> list[str]:
    """The keys the paper's sources use, or, without those sources, the keys of the committed
    ``numbers.tex``."""
    if not (root / PAPER_DIR / "sections").is_dir():
        committed = (root / GENERATED_DIR / "numbers.tex").read_text(encoding="utf-8")
        return sorted(set(NAMEDEF_RE.findall(committed)))
    keys: set[str] = set()
    for path in source_files(root):
        for key in USE_RE.findall(strip_comments(path.read_text(encoding="utf-8"))):
            if not KEY_RE.fullmatch(key):
                raise ValueError(f"{path.name}: the key {key!r} holds a character TeX would misread")
            keys.add(key)
    return sorted(keys)


def render_numbers(pool: dict[str, str], keys: Iterable[str]) -> str:
    keys = list(keys)
    missing = [k for k in keys if k not in pool]
    if missing:
        raise KeyError(f"the paper uses keys the pool does not hold: {missing}")
    lines = [
        "\\makeatletter",
        "\\DeclareRobustCommand{\\crn}[1]{\\@ifundefined{crn@#1}{\\PackageError{pcpbench}"
        "{unknown number key #1}{Run scripts/camera_ready_numbers.py.}}{\\@nameuse{crn@#1}}}",
    ]
    lines += [f"\\@namedef{{crn@{key}}}{{{pool[key]}}}" for key in keys]
    lines.append("\\makeatother")
    return "\n".join(lines) + "\n"


def _row(cells: Iterable[str]) -> str:
    return " & ".join(cells) + " \\\\"


def _math(text: str) -> str:
    return f"${text}$"


def tab_reference(root: Path) -> str:
    """K, the points per class, the matched reference (mean and single-draw sd) and the oracle
    the submission reported at that setting."""
    rows = [r for r in _load(root, VALIDITY_FLOOR)["rows"] if r["setting"] in ("k_sweep", "imagenet1k")]
    setting = {"k_sweep": "CIFAR-100 sweep", "imagenet1k": "ImageNet-1k"}
    lines = []
    for r in rows:
        lines.append(_row([setting[r["setting"]], str(r["K"]), f"{r['n_cal']}/{r['n_eval']}",
                           _math(f3(r["floor_mean"])), _math(f3(r["floor_sd"])),
                           _math(f3(r["paper_oracle"]))]))
    return "\n".join(lines) + "\n"


def tab_perk(root: Path) -> str:
    """Reference and mean worst-class coverage over the five selected caches."""
    trend = _load(root, ALL_CACHE_TREND)
    reference = [trend["reference"][str(k)]["mean"] for k in K_VALUES]
    index = {(r["method"], r["k"]): r for r in trend["scopes"][MODERATE_SCOPE]["per_k"]}
    lines = [_row(["Matched reference", *(_math(dot3(x)) for x in reference)]), "\\midrule"]
    for method in HEADLINE_ORDER:
        lines.append(_row([METHOD_NAME[method],
                           *(_mean_cell(index[(method, k)]) for k in K_VALUES)]))
    return "\n".join(lines) + "\n"


def _mean_cell(row: dict) -> str:
    return _math(dot3(row["mean"]))


def _trend_cell(row: dict) -> str:
    return f"{_math(dot3(row['mean']))} {{\\scriptsize({dot2(row['zero_share'])})}}"


def tab_allcache(root: Path, methods: Iterable[str] = HEADLINE_ORDER) -> str:
    """Every cache of each backbone: mean worst-class coverage at every K, with the share of
    (cache, seed) cells that read exactly zero in parentheses."""
    trend = _load(root, ALL_CACHE_TREND)
    reference = [trend["reference"][str(k)]["mean"] for k in K_VALUES]
    lines = [_row(["Matched reference", *(_math(dot3(x)) for x in reference),
                   *(_math(dot3(x)) for x in reference)]), "\\midrule"]
    index = {(scope, r["method"], r["k"]): r
             for scope, block in trend["scopes"].items() for r in block["per_k"]}
    for method in methods:
        cells = [_trend_cell(index[(scope, method, k)]) for scope in SCOPE_SLUG for k in K_VALUES]
        lines.append(_row([METHOD_NAME[method], *cells]))
    return "\n".join(lines) + "\n"


def tab_clustered(root: Path) -> str:
    """Clustered means on the selected caches and zero shares on the full grid."""
    trend = _load(root, ALL_CACHE_TREND)
    reference = [trend["reference"][str(k)]["mean"] for k in K_VALUES]
    lines = [_row(["Reference", *(_math(dot3(x)) for x in reference)]), "\\midrule"]
    index = {(scope, r["k"]): r for scope, block in trend["scopes"].items()
             for r in block["per_k"] if r["method"] == SUPPLEMENTARY}
    lines.append(_row(["Five moderate",
                       *(_mean_cell(index[(MODERATE_SCOPE, k)]) for k in K_VALUES)]))
    for scope in SCOPE_SLUG:
        lines.append(_row([f"All {scope}",
                           *(_trend_cell(index[(scope, k)]) for k in K_VALUES)]))
    return "\n".join(lines) + "\n"


def tab_nested(root: Path) -> str:
    """Mean nested contributions on the selected caches and without shift."""
    rows = {r["method"]: r
            for r in _load(root, NESTED_TREND)["scopes"][MODERATE_SCOPE]["decomposition"]}
    lines = []
    for method in HEADLINE_ORDER:
        cells = [METHOD_NAME[method]]
        for term in FAMILY_TERMS:
            stat = rows[method][term]
            cells += [_math(sdot3(stat["mean"])), _math(sdot3(stat["no_shift"]["mean"]))]
        lines.append(_row(cells))
    return "\n".join(lines) + "\n"


def _board_tables(root: Path) -> tuple[dict, list]:
    al = registry_module.analyze_module(root)
    board = registry_module.rerun_board(root, al)
    return registry_module._table_values(board, al), board


def tab_delta(root: Path) -> str:
    table = _board_tables(root)[0]["tab:delta"]
    return "\n".join(_row([METHOD_NAME[m], table[m][0], _math(table[m][1]), _math(table[m][2])])
                     for m in DELTA_ORDER) + "\n"


def tab_vitl(root: Path) -> str:
    table = _board_tables(root)[0]["tab:vitl"]
    return "\n".join(_row([METHOD_NAME[m], _math(table[m][0]), _math(table[m][1])])
                     for m in VITL_ORDER) + "\n"


def tab_imagenet(root: Path) -> str:
    tables, board = _board_tables(root)
    lines = [_row([METHOD_NAME[m], *(_math(c) for c in tables["tab:imagenet"][m])])
             for m in IMAGENET_ORDER]
    lines.append("\\midrule")
    lines.append(_row([f"{METHOD_NAME[SUPPLEMENTARY]} (supplementary)",
                       *(_math(c) for c in _imagenet_columns(board, SUPPLEMENTARY))]))
    return "\n".join(lines) + "\n"


TABLES: dict[str, Callable[[Path], str]] = {
    "tab_reference": tab_reference,
    "tab_perk": tab_perk,
    "tab_allcache": tab_allcache,
    "tab_delta": tab_delta,
    "tab_imagenet": tab_imagenet,
    "tab_vitl": tab_vitl,
    "tab_clustered": tab_clustered,
    "tab_nested": tab_nested,
}


def render_all(root: Path = _REPO_ROOT) -> dict[str, str]:
    """File name under ``generated/`` -> its content."""
    root = Path(root)
    out = {"numbers.tex": render_numbers(build_pool(root), used_keys(root))}
    for name, render in TABLES.items():
        out[f"{name}.tex"] = render(root)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 when a generated file differs from a fresh render")
    args = ap.parse_args(argv)
    rendered = render_all(args.root)
    target = args.root / GENERATED_DIR
    if args.check:
        stale = [name for name, text in rendered.items()
                 if not (target / name).exists() or (target / name).read_text(encoding="utf-8") != text]
        extra = sorted(p.name for p in target.glob("*.tex") if p.name not in rendered)
        for name in stale:
            print(f"STALE: {GENERATED_DIR}/{name}")
        for name in extra:
            print(f"UNEXPECTED: {GENERATED_DIR}/{name}")
        print("CHECK FAILED" if stale or extra else f"CHECK OK: {len(rendered)} generated files")
        return 1 if stale or extra else 0
    target.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (target / name).write_text(text, encoding="utf-8")
        print(f"wrote {GENERATED_DIR}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
