#!/usr/bin/env python3
"""Leaderboard-analysis harness for the pcpbench conformal-prediction benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple


_SEPARATOR_CELL_RE = re.compile(r":?-{2,}:?")
_STRUCTURED_CIFAR_RE = re.compile(r"_(?P<corruption>[a-z0-9]+(?:_[a-z0-9]+)*)_s(?P<severity>\d+)$")
_CIFAR_FAMILIES = ("cifar100c", "cifar10c")
SCORES: tuple[str, ...] = ("lac", "aps")
_NON_LAC_SUFFIXES = tuple(f"_{score}" for score in SCORES if score != "lac")
HEADLINE_METHODS = frozenset({"split", "mondrian", "weighted", "ppi", "acp", "oracle"})


class DatasetInfo(NamedTuple):
    """Grammar-parsed fields derived from a raw ``dataset`` cell."""

    family: str
    corruption: str | None
    severity: int | None
    backbone: str | None
    structured: bool


class Row(NamedTuple):
    """One parsed data row of ``leaderboard.md``, plus derived fields."""

    dataset: str
    method: str
    marginal: float
    worst_class: float
    frac_undercovered: float
    mean_set_size: float
    set_over_k: float
    demoted: bool
    family: str
    corruption: str | None
    severity: int | None
    backbone: str | None
    structured: bool
    k: int | None


def parse_dataset_name(name: str) -> DatasetInfo:
    """Parse a raw ``dataset`` cell into family/corruption/severity/backbone."""
    for prefix in _CIFAR_FAMILIES:
        if not name.startswith(prefix):
            continue
        backbone = "ViT-L" if "_vitL" in name else "ViT-B"
        remainder = name[len(prefix):].replace("_vitL", "")
        match = _STRUCTURED_CIFAR_RE.search(remainder) if remainder.startswith("_") else None
        if match and match.group(0) == remainder:
            return DatasetInfo(
                family=prefix,
                corruption=match.group("corruption"),
                severity=int(match.group("severity")),
                backbone=backbone,
                structured=True,
            )
        return DatasetInfo(family=prefix, corruption=None, severity=None, backbone=backbone, structured=False)
    return DatasetInfo(family=name, corruption=None, severity=None, backbone=None, structured=False)


_FAMILY_K: dict[str, int] = {
    "cifar10c": 10,
    "cifar100c": 100,
    "ntu60_xsub": 60,
    "ntu2nucla": 5,
}

_EXPECTED_VITB_KSCALING_CELLS = 40


def _recover_k(mean_set_size: float, set_over_k: float) -> int | None:
    if set_over_k <= 0:
        return None
    return round(mean_set_size / set_over_k)


def _class_count(family: str, mean_set_size: float, set_over_k: float) -> int | None:
    if family in _FAMILY_K:
        return _FAMILY_K[family]
    return _recover_k(mean_set_size, set_over_k)


def _is_divider_row(cells: list[str]) -> bool:
    return len(cells) > 1 and bool(cells[0]) and not any(c for c in cells[1:])


def _is_separator_row(cells: list[str]) -> bool:
    return all(_SEPARATOR_CELL_RE.fullmatch(c) for c in cells if c) and any(cells)


_PARSED_COLUMNS = (
    "dataset", "method", "marginal", "worst_class", "frac_undercovered",
    "mean_set_size", "set/K",
)


def _column_indices(header_cells: list[str]) -> tuple[int, ...]:
    names = [c.strip() for c in header_cells]
    missing = [name for name in _PARSED_COLUMNS if name not in names]
    if missing:
        raise ValueError(
            f"leaderboard header lacks required column(s) {missing}; got {names}"
        )
    return tuple(names.index(name) for name in _PARSED_COLUMNS)


def parse_leaderboard(text: str) -> list[Row]:
    """Parse a ``leaderboard.md`` pipe-table into a list of ``Row``."""
    rows: list[Row] = []
    demoted = False
    pipe_row_index = 0
    indices = tuple(range(len(_PARSED_COLUMNS)))
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("|"):
            if line.startswith("_set/K") or line.startswith("_set/k"):
                demoted = True
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        pipe_row_index += 1
        if pipe_row_index == 1:
            indices = _column_indices(cells)
            continue
        if _is_separator_row(cells):
            continue
        if _is_divider_row(cells):
            demoted = True
            continue
        if len(cells) <= max(indices):
            continue
        dataset, method, marginal_s, worst_s, frac_s, mss_s, sk_s = (cells[i] for i in indices)
        try:
            marginal, worst, frac, mss, set_over_k = (
                float(marginal_s), float(worst_s), float(frac_s), float(mss_s), float(sk_s),
            )
        except ValueError:
            continue
        info = parse_dataset_name(dataset)
        rows.append(Row(
            dataset=dataset, method=method, marginal=marginal, worst_class=worst,
            frac_undercovered=frac, mean_set_size=mss, set_over_k=set_over_k, demoted=demoted,
            family=info.family, corruption=info.corruption, severity=info.severity,
            backbone=info.backbone, structured=info.structured,
            k=_class_count(info.family, mss, set_over_k),
        ))
    return rows


TRIVIAL_VACUOUS_WORST = 0.99
TRIVIAL_VACUOUS_SETK = 0.90


def is_trivially_vacuous(worst_class: float, set_over_k: float) -> bool:
    """A cell whose set is nearly all classes AND that therefore trivially covers every class
    (worst ~= 1.0)."""
    return worst_class >= TRIVIAL_VACUOUS_WORST and set_over_k >= TRIVIAL_VACUOUS_SETK


class KScalingCell(NamedTuple):
    method: str
    corruption: str
    severity: int
    backbone: str
    low_k: int
    high_k: int
    marginal_low_k: float
    marginal_high_k: float
    worst_class_low_k: float
    worst_class_high_k: float
    delta_worst: float
    informative: bool


class KScalingSummary(NamedTuple):
    method: str
    n_cells_all: int
    mean_delta_worst_all: float
    n_cells_informative: int
    mean_delta_worst_informative: float | None


def exhibit_k_scaling(rows: list[Row]) -> tuple[list[KScalingCell], list[KScalingSummary]]:
    """Match structured rows sharing (method, corruption, severity, backbone) across distinct
    recovered K values, and report how ``worst_class`` changes from the lowest to the highest K
    in each matched group."""
    groups: dict[tuple[str, str, int, str], dict[int, Row]] = defaultdict(dict)
    for row in rows:
        if not row.structured or row.corruption is None or row.severity is None:
            continue
        if row.backbone is None or row.k is None:
            continue
        key = (row.method, row.corruption, row.severity, row.backbone)
        groups[key].setdefault(row.k, row)

    cells: list[KScalingCell] = []
    for (method, corruption, severity, backbone), by_k in groups.items():
        if len(by_k) < 2:
            continue
        low_k, high_k = min(by_k), max(by_k)
        low_row, high_row = by_k[low_k], by_k[high_k]
        informative = not (
            is_trivially_vacuous(low_row.worst_class, low_row.set_over_k)
            or is_trivially_vacuous(high_row.worst_class, high_row.set_over_k)
        )
        cells.append(KScalingCell(
            method=method, corruption=corruption, severity=severity, backbone=backbone,
            low_k=low_k, high_k=high_k,
            marginal_low_k=low_row.marginal, marginal_high_k=high_row.marginal,
            worst_class_low_k=low_row.worst_class, worst_class_high_k=high_row.worst_class,
            delta_worst=high_row.worst_class - low_row.worst_class,
            informative=informative,
        ))
    cells.sort(key=lambda c: (c.method, c.corruption, c.severity, c.backbone))

    cells_by_method: dict[str, list[KScalingCell]] = defaultdict(list)
    for cell in cells:
        cells_by_method[cell.method].append(cell)

    for method, method_cells in cells_by_method.items():
        n_vitb = sum(1 for c in method_cells if c.backbone == "ViT-B")
        if n_vitb and n_vitb != _EXPECTED_VITB_KSCALING_CELLS:
            print(
                f"WARNING: K-scaling for method {method!r} has {n_vitb} ViT-B "
                f"matched pairs, expected {_EXPECTED_VITB_KSCALING_CELLS} "
                f"(8 corruptions x 5 severities). A canonical pair may be missing.",
                file=sys.stderr,
            )

    summaries: list[KScalingSummary] = []
    for method, method_cells in sorted(cells_by_method.items()):
        deltas_all = [c.delta_worst for c in method_cells]
        deltas_informative = [c.delta_worst for c in method_cells if c.informative]
        summaries.append(KScalingSummary(
            method=method,
            n_cells_all=len(deltas_all),
            mean_delta_worst_all=sum(deltas_all) / len(deltas_all),
            n_cells_informative=len(deltas_informative),
            mean_delta_worst_informative=(
                sum(deltas_informative) / len(deltas_informative) if deltas_informative else None
            ),
        ))
    return cells, summaries


RECOVERED = "recovered"
REGIME_A = "A_marginal_holds_perclass_collapses"
REGIME_B = "B_marginal_breaks"
REGIME_OTHER = "other"


def label_regime(
    marginal: float,
    worst_class: float,
    *,
    recovered: float = 0.80,
    regime_a_marginal: float = 0.85,
    regime_b_marginal: float = 0.80,
) -> str:
    """Classify a single (marginal, worst_class) pair into one of the four two-regime labels."""
    if worst_class >= recovered:
        return RECOVERED
    if marginal >= regime_a_marginal and worst_class < recovered:
        return REGIME_A
    if marginal < regime_b_marginal:
        return REGIME_B
    return REGIME_OTHER


class FamilyMethodCount(NamedTuple):
    family: str
    method: str
    label: str
    count: int


class RegimeExhibit(NamedTuple):
    overall_counts: dict[str, int]
    demoted_overall_counts: dict[str, int]
    per_family_method: list[FamilyMethodCount]


def exhibit_regime_labels(rows: list[Row]) -> RegimeExhibit:
    """Aggregate two-regime labels."""
    non_demoted = [row for row in rows if not row.demoted]
    demoted = [row for row in rows if row.demoted]

    overall = Counter(label_regime(row.marginal, row.worst_class) for row in non_demoted)
    demoted_overall = Counter(label_regime(row.marginal, row.worst_class) for row in demoted)

    per_group: Counter[tuple[str, str, str]] = Counter()
    for row in non_demoted:
        label = label_regime(row.marginal, row.worst_class)
        per_group[(row.family, row.method, label)] += 1
    per_family_method = sorted(
        (FamilyMethodCount(family, method, label, count) for (family, method, label), count in per_group.items()),
        key=lambda item: (item.family, item.method, item.label),
    )
    return RegimeExhibit(
        overall_counts=dict(overall),
        demoted_overall_counts=dict(demoted_overall),
        per_family_method=per_family_method,
    )


EXEMPT_CORRUPTIONS = frozenset({"saturate", "fog", "elastic_transform", "spatter_mud"})
MONOTONICITY_TOLERANCE = 0.03


class MonotonicityViolation(NamedTuple):
    family: str
    corruption: str
    method: str
    severity_low: int
    severity_high: int
    worst_class_low: float
    worst_class_high: float


def exhibit_monotonicity(rows: list[Row]) -> tuple[list[MonotonicityViolation], int]:
    """Flag (family, corruption, method) triples -- with >= 3 of severities 1..5 present and
    corruption not in ``EXEMPT_CORRUPTIONS`` -- where ``worst_class`` rises by more than
    ``MONOTONICITY_TOLERANCE`` between consecutive present severities."""
    groups: dict[tuple[str, str, str], dict[int, Row]] = defaultdict(dict)
    for row in rows:
        if not row.structured or row.corruption is None or row.severity is None:
            continue
        if row.corruption in EXEMPT_CORRUPTIONS:
            continue
        groups[(row.family, row.corruption, row.method)][row.severity] = row

    violations: list[MonotonicityViolation] = []
    n_dropped = 0
    for (family, corruption, method), by_severity in groups.items():
        in_range = {sev: row for sev, row in by_severity.items() if 1 <= sev <= 5}
        kept = {
            sev: row for sev, row in in_range.items()
            if not is_trivially_vacuous(row.worst_class, row.set_over_k)
        }
        n_dropped += len(in_range) - len(kept)
        present = sorted(kept)
        if len(present) < 3:
            continue
        for lo, hi in zip(present, present[1:]):
            worst_lo, worst_hi = kept[lo].worst_class, kept[hi].worst_class
            if worst_hi > worst_lo + MONOTONICITY_TOLERANCE:
                violations.append(MonotonicityViolation(family, corruption, method, lo, hi, worst_lo, worst_hi))
    violations.sort(key=lambda v: (v.family, v.corruption, v.method, v.severity_low))
    return violations, n_dropped


def _render_k_scaling_md(cells: list[KScalingCell], summaries: list[KScalingSummary]) -> str:
    lines = ["## Exhibit 1: K-scaling (per-class validity decay)", ""]
    if not cells:
        lines.append("No matched (method, corruption, severity, backbone) groups across >=2 K values.")
        return "\n".join(lines) + "\n"
    header = (
        "| method | corruption | severity | backbone | low_k | high_k | marginal@lowK | marginal@highK | "
        "worst@lowK | worst@highK | delta_worst | informative |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines += [header, sep]
    for c in cells:
        lines.append(
            f"| {c.method} | {c.corruption} | {c.severity} | {c.backbone} | {c.low_k} | {c.high_k} | "
            f"{c.marginal_low_k:.3f} | {c.marginal_high_k:.3f} | {c.worst_class_low_k:.3f} | "
            f"{c.worst_class_high_k:.3f} | {c.delta_worst:+.3f} | {c.informative} |"
        )
    lines += [
        "",
        "### Per-method mean delta_worst (worst@highK - worst@lowK)",
        "",
        "_all = every matched cell. informative = excludes cells where either K endpoint is "
        "trivially-vacuous (near-full prediction set that trivially covers every class, worst~=1.0) -- "
        "a saturation artifact, not real per-class recovery/collapse. The informative acp value is the "
        "headline._",
        "",
    ]
    lines += [
        "| method | n_cells_all | mean_delta_worst_all | n_cells_informative | mean_delta_worst_informative |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        marker = "  <-- headline (acp)" if s.method == "acp" else ""
        informative_str = (
            f"{s.mean_delta_worst_informative:+.3f}" if s.mean_delta_worst_informative is not None else "n/a"
        )
        lines.append(
            f"| {s.method} | {s.n_cells_all} | {s.mean_delta_worst_all:+.3f} | "
            f"{s.n_cells_informative} | {informative_str} |{marker}"
        )
    if not any(s.method == "acp" for s in summaries):
        lines.append("\n_(no acp cells matched across K -- acp headline unavailable for this leaderboard.)_")
    return "\n".join(lines) + "\n"


def _render_regime_md(regime: RegimeExhibit) -> str:
    lines = ["## Exhibit 2: two-regime labeling", "", "### Overall counts (non-demoted rows)", ""]
    lines += ["| label | count |", "| --- | --- |"]
    for label in (RECOVERED, REGIME_A, REGIME_B, REGIME_OTHER):
        lines.append(f"| {label} | {regime.overall_counts.get(label, 0)} |")
    lines += ["", "### Overall counts (demoted rows)", ""]
    lines += ["| label | count |", "| --- | --- |"]
    for label in (RECOVERED, REGIME_A, REGIME_B, REGIME_OTHER):
        lines.append(f"| {label} | {regime.demoted_overall_counts.get(label, 0)} |")
    lines += ["", "### Per (family, method) counts (non-demoted rows)", ""]
    lines += ["| family | method | label | count |", "| --- | --- | --- | --- |"]
    for row in regime.per_family_method:
        lines.append(f"| {row.family} | {row.method} | {row.label} | {row.count} |")
    return "\n".join(lines) + "\n"


def _render_monotonicity_md(violations: list[MonotonicityViolation], n_severities_dropped_saturated: int) -> str:
    lines = ["## Exhibit 3: severity-monotonicity QC", ""]
    lines.append(
        f"_{n_severities_dropped_saturated} severity point(s) dropped as trivially-vacuous "
        "(saturated near-full prediction set) before flagging -- see `is_trivially_vacuous`._"
    )
    lines.append("")
    if not violations:
        lines.append("Clean: no monotonicity violations found.")
        return "\n".join(lines) + "\n"
    lines += [
        "| family | corruption | method | severity_low | severity_high | worst@low | worst@high |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for v in violations:
        lines.append(
            f"| {v.family} | {v.corruption} | {v.method} | {v.severity_low} | {v.severity_high} | "
            f"{v.worst_class_low:.3f} | {v.worst_class_high:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_report(
    n_rows: int,
    k_cells: list[KScalingCell],
    k_summaries: list[KScalingSummary],
    regime: RegimeExhibit,
    violations: list[MonotonicityViolation],
    n_severities_dropped_saturated: int,
) -> str:
    parts = [
        "# pcpbench leaderboard analysis",
        "",
        f"Parsed {n_rows} data rows.",
        "",
        _render_k_scaling_md(k_cells, k_summaries),
        _render_regime_md(regime),
        _render_monotonicity_md(violations, n_severities_dropped_saturated),
    ]
    return "\n".join(parts)


def _build_json_payload(
    leaderboard_path: str,
    rows: list[Row],
    k_cells: list[KScalingCell],
    k_summaries: list[KScalingSummary],
    regime: RegimeExhibit,
    violations: list[MonotonicityViolation],
    n_severities_dropped_saturated: int,
) -> dict:
    return {
        "leaderboard_path": leaderboard_path,
        "n_rows": len(rows),
        "k_scaling": {
            "cells": [cell._asdict() for cell in k_cells],
            "summary": [s._asdict() for s in k_summaries],
        },
        "regime": {
            "overall_counts": regime.overall_counts,
            "demoted_overall_counts": regime.demoted_overall_counts,
            "per_family_method": [item._asdict() for item in regime.per_family_method],
        },
        "monotonicity": {
            "violations": [v._asdict() for v in violations],
            "n_severities_dropped_saturated": n_severities_dropped_saturated,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_leaderboard",
        description="Compute K-scaling, two-regime labeling, and severity-monotonicity exhibits from leaderboard.md.",
    )
    parser.add_argument("--leaderboard", default="leaderboard.md", help="Path to the leaderboard.md pipe-table.")
    parser.add_argument(
        "--out", default="outputs/analysis/leaderboard_analysis.json", help="Path to write the JSON report to."
    )
    parser.add_argument(
        "--subset",
        choices=("all", "headline"),
        default="all",
        help=(
            "Which rows to analyze. 'all' (default) uses every row in the file. "
            "'headline' selects the paper's ViT-B/32 headline board (the 82 shift "
            "conditions: structured cifar ViT-B/32 corruptions plus the two skeleton "
            "conditions), dropping ViT-L/14, ImageNet-1k, and the unstructured cifar "
            "aliases. Use 'headline' to reproduce "
            "outputs/analysis/leaderboard_analysis.json from the full released "
            "leaderboard.md."
        ),
    )
    return parser


def select_headline(rows: list[Row]) -> list[Row]:
    """The paper's ViT-B/32 headline board: the 82 shift conditions."""
    return [
        row
        for row in rows
        if ((row.structured and row.backbone == "ViT-B") or row.family.startswith("ntu"))
        and row.method in HEADLINE_METHODS
        and not row.method.endswith(_NON_LAC_SUFFIXES)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    leaderboard_path = Path(args.leaderboard)
    text = leaderboard_path.read_text()
    rows = parse_leaderboard(text)

    if args.subset == "headline":
        rows = select_headline(rows)
        if not rows:
            parser.error("--subset headline matched no rows in the leaderboard.")

    k_cells, k_summaries = exhibit_k_scaling(rows)
    regime = exhibit_regime_labels(rows)
    violations, n_dropped = exhibit_monotonicity(rows)

    report = render_report(len(rows), k_cells, k_summaries, regime, violations, n_dropped)
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_json_payload(
        str(leaderboard_path), rows, k_cells, k_summaries, regime, violations, n_dropped
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
