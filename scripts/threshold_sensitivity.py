#!/usr/bin/env python3
"""Regime and guard threshold sensitivity over the pcpbench headline leaderboard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from pcpbench.leaderboard import (
    OVERCOVER_MARGINAL,
    VACUOUS_OVERCOVER_FRAC,
    VACUOUS_SET_FRAC,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYZE_LEADERBOARD_PATH = _SCRIPT_DIR / "analyze_leaderboard.py"


def _load_analyze_leaderboard():
    spec = importlib.util.spec_from_file_location("analyze_leaderboard", _ANALYZE_LEADERBOARD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


al = _load_analyze_leaderboard()

RECOVERED = al.RECOVERED
REGIME_A = al.REGIME_A
REGIME_B = al.REGIME_B
REGIME_OTHER = al.REGIME_OTHER


DEFAULT_RECOVERED = 0.80
DEFAULT_REGIME_A_MARGINAL = 0.85
DEFAULT_REGIME_B_MARGINAL = 0.80

EXPECTED_DEFAULT_COUNTS: dict[str, int] = {
    "non_demoted": 404,
    "recovered": 86,
    "regime_a": 110,
    "regime_b": 194,
    "other": 14,
    "demoted": 88,
    "demoted_recovered": 60,
    "demoted_regime_a": 28,
}


REGIME_GRID_RECOVERED = (0.75, 0.80, 0.85)
REGIME_GRID_A_MARGINAL = (0.80, 0.85, 0.90)
REGIME_GRID_B_MARGINAL = (0.75, 0.80, 0.85)

GUARD_GRID_SET_FRAC = (0.85, 0.90, 0.95)
GUARD_GRID_OVERCOVER_PAIRS = ((0.50, 0.95), (0.60, 0.95), (0.70, 0.95), (0.60, 0.90))

_COUNT_KEYS = ("non_demoted", "recovered", "regime_a", "regime_b", "other", "demoted")


def is_vacuous_row(
    marginal: float,
    set_over_k: float,
    *,
    vacuous_set_frac: float = VACUOUS_SET_FRAC,
    vacuous_overcover_frac: float = VACUOUS_OVERCOVER_FRAC,
    overcover_marginal: float = OVERCOVER_MARGINAL,
) -> bool:
    """Whether a row with this ``marginal`` and ``set_over_k`` (``set/K``, already parsed from
    ``leaderboard.md``) would be demoted."""
    return set_over_k >= vacuous_set_frac or (
        set_over_k >= vacuous_overcover_frac and marginal >= overcover_marginal
    )


def is_vacuous_row_min(
    marginal: float,
    set_over_k: float,
    *,
    vacuous_set_frac: float = VACUOUS_SET_FRAC,
    vacuous_overcover_frac: float = VACUOUS_OVERCOVER_FRAC,
    overcover_marginal: float = OVERCOVER_MARGINAL,
) -> bool:
    """Lower bound on ``is_vacuous_row`` when ``marginal``/``set_over_k`` are values
    ``leaderboard.md`` displays (``set/K`` to 2 decimals, ``marginal`` to 3) rather than the
    unrounded values a row was computed from."""
    return set_over_k > vacuous_set_frac or (
        set_over_k > vacuous_overcover_frac and marginal > overcover_marginal
    )


def _is_default_guard_combo(combo: "GuardCombo") -> bool:
    return (
        math.isclose(combo.vacuous_set_frac, VACUOUS_SET_FRAC, abs_tol=1e-9)
        and math.isclose(combo.vacuous_overcover_frac, VACUOUS_OVERCOVER_FRAC, abs_tol=1e-9)
        and math.isclose(combo.overcover_marginal, OVERCOVER_MARGINAL, abs_tol=1e-9)
    )


def _is_ambiguous_row(marginal: float, set_over_k: float, combo: "GuardCombo") -> bool:
    return (
        math.isclose(set_over_k, combo.vacuous_set_frac, abs_tol=1e-9)
        or math.isclose(set_over_k, combo.vacuous_overcover_frac, abs_tol=1e-9)
        or math.isclose(marginal, combo.overcover_marginal, abs_tol=1e-9)
    )


class MissingUnroundedRowError(RuntimeError):
    """Raised by ``load_unrounded_guard_inputs`` when ``--outputs-dir`` is given but a headline
    row has no unrounded row JSON under it."""


def load_unrounded_guard_inputs(rows: list, outputs_dir: Path) -> list[tuple[float, float]]:
    """For every row in ``rows``, read its unrounded ``marginal``, ``mean_set_size`` and
    ``n_classes`` from ``<outputs_dir>/<dataset>_<method>.json`` (the row JSON ``pcpbench run
    --out`` writes) and return the ``(marginal, mean_set_size / n_classes)`` pairs the guard
    needs, in ``rows`` order."""
    inputs: list[tuple[float, float]] = []
    for row in rows:
        path = outputs_dir / f"{row.dataset}_{row.method}.json"
        if not path.exists():
            raise MissingUnroundedRowError(
                f"--outputs-dir {outputs_dir}: no unrounded row JSON for headline row "
                f"{row.dataset}/{row.method} (expected {path})"
            )
        data = json.loads(path.read_text())
        marginal = float(data["marginal"])
        set_over_k = float(data["mean_set_size"]) / float(data["n_classes"])
        inputs.append((marginal, set_over_k))
    return inputs


class RegimeCombo(NamedTuple):
    recovered: float
    regime_a_marginal: float
    regime_b_marginal: float


class GuardCombo(NamedTuple):
    vacuous_set_frac: float
    vacuous_overcover_frac: float
    overcover_marginal: float


def default_row_status(row) -> tuple[bool, str]:
    """The row's (demoted, label) at the single shared default combination, using the file's own
    ``demoted`` flag as ground truth."""
    return (row.demoted, al.label_regime(row.marginal, row.worst_class))


def regime_grid_status(row, combo: RegimeCombo) -> tuple[bool, str]:
    label = al.label_regime(
        row.marginal, row.worst_class,
        recovered=combo.recovered,
        regime_a_marginal=combo.regime_a_marginal,
        regime_b_marginal=combo.regime_b_marginal,
    )
    return (row.demoted, label)


def guard_grid_status(row, combo: GuardCombo) -> tuple[bool, str]:
    demoted = is_vacuous_row(
        row.marginal, row.set_over_k,
        vacuous_set_frac=combo.vacuous_set_frac,
        vacuous_overcover_frac=combo.vacuous_overcover_frac,
        overcover_marginal=combo.overcover_marginal,
    )
    label = al.label_regime(row.marginal, row.worst_class)
    return (demoted, label)


def _counts_from_statuses(statuses: list[tuple[bool, str]]) -> dict[str, int]:
    non_demoted_labels = Counter(label for demoted, label in statuses if not demoted)
    demoted_labels = Counter(label for demoted, label in statuses if demoted)
    return {
        "non_demoted": sum(non_demoted_labels.values()),
        "recovered": non_demoted_labels.get(RECOVERED, 0),
        "regime_a": non_demoted_labels.get(REGIME_A, 0),
        "regime_b": non_demoted_labels.get(REGIME_B, 0),
        "other": non_demoted_labels.get(REGIME_OTHER, 0),
        "demoted": sum(demoted_labels.values()),
    }


def reproduce_default_counts(rows: list) -> dict[str, int]:
    """The six paper counts plus the two "before demotion" counts, computed straight from the
    file's own ``demoted`` flag and the default ``label_regime`` thresholds -- no guard
    recomputation, no rounding risk."""
    statuses = [default_row_status(r) for r in rows]
    counts = _counts_from_statuses(statuses)
    demoted_labels = Counter(label for demoted, label in statuses if demoted)
    counts["demoted_recovered"] = demoted_labels.get(RECOVERED, 0)
    counts["demoted_regime_a"] = demoted_labels.get(REGIME_A, 0)
    return counts


def regime_grid_counts(rows: list, combo: RegimeCombo) -> dict[str, int]:
    return _counts_from_statuses([regime_grid_status(r, combo) for r in rows])


def guard_grid_counts(rows: list, combo: GuardCombo) -> dict[str, int]:
    return _counts_from_statuses([guard_grid_status(r, combo) for r in rows])


def _rows_relabelled(current_statuses: list[tuple[bool, str]], default_statuses: list[tuple[bool, str]]) -> int:
    return sum(1 for cur, base in zip(current_statuses, default_statuses) if cur != base)


def build_regime_grid(rows: list) -> list[dict]:
    default_statuses = [default_row_status(r) for r in rows]
    records = []
    for recovered in REGIME_GRID_RECOVERED:
        for regime_a_marginal in REGIME_GRID_A_MARGINAL:
            for regime_b_marginal in REGIME_GRID_B_MARGINAL:
                combo = RegimeCombo(recovered, regime_a_marginal, regime_b_marginal)
                statuses = [regime_grid_status(r, combo) for r in rows]
                records.append({
                    "recovered": recovered,
                    "regime_a_marginal": regime_a_marginal,
                    "regime_b_marginal": regime_b_marginal,
                    "counts": _counts_from_statuses(statuses),
                    "rows_relabelled": _rows_relabelled(statuses, default_statuses),
                })
    return records


def build_guard_grid(rows: list, outputs_dir: Path | None = None) -> tuple[list[dict], str]:
    """The guard threshold grid."""
    default_statuses = [default_row_status(r) for r in rows]
    guard_inputs = load_unrounded_guard_inputs(rows, outputs_dir) if outputs_dir is not None else None
    precision = "unrounded" if outputs_dir is not None else "displayed"

    records = []
    for vacuous_set_frac in GUARD_GRID_SET_FRAC:
        for vacuous_overcover_frac, overcover_marginal in GUARD_GRID_OVERCOVER_PAIRS:
            combo = GuardCombo(vacuous_set_frac, vacuous_overcover_frac, overcover_marginal)
            from_values: list[bool] = []
            from_values_min: list[bool] = []
            on_threshold: list[bool] = []
            for i, row in enumerate(rows):
                if guard_inputs is not None:
                    val_marginal, val_set_over_k = guard_inputs[i]
                else:
                    val_marginal, val_set_over_k = row.marginal, row.set_over_k

                demoted = is_vacuous_row(
                    val_marginal, val_set_over_k,
                    vacuous_set_frac=vacuous_set_frac,
                    vacuous_overcover_frac=vacuous_overcover_frac,
                    overcover_marginal=overcover_marginal,
                )
                from_values.append(demoted)
                if guard_inputs is None:
                    on_threshold.append(_is_ambiguous_row(val_marginal, val_set_over_k, combo))
                    from_values_min.append(is_vacuous_row_min(
                        val_marginal, val_set_over_k,
                        vacuous_set_frac=vacuous_set_frac,
                        vacuous_overcover_frac=vacuous_overcover_frac,
                        overcover_marginal=overcover_marginal,
                    ))
                else:
                    on_threshold.append(False)
                    from_values_min.append(demoted)

            displayed_only = [
                f"{row.dataset}/{row.method}"
                for row, demoted in zip(rows, from_values)
                if demoted and not row.demoted
            ]
            divider_settled = (
                guard_inputs is None
                and _is_default_guard_combo(combo)
                and all(
                    demoted == row.demoted or (demoted and not row.demoted and not strict)
                    for row, demoted, strict in zip(rows, from_values, from_values_min)
                )
            )
            demotions = [row.demoted for row in rows] if divider_settled else from_values
            statuses = [
                (demoted, al.label_regime(row.marginal, row.worst_class))
                for row, demoted in zip(rows, demotions)
            ]
            counts = _counts_from_statuses(statuses)

            records.append({
                "vacuous_set_frac": vacuous_set_frac,
                "vacuous_overcover_frac": vacuous_overcover_frac,
                "overcover_marginal": overcover_marginal,
                "counts": counts,
                "rows_relabelled": _rows_relabelled(statuses, default_statuses),
                "demoted_min": counts["demoted"] if divider_settled else sum(from_values_min),
                "demoted_max": counts["demoted"],
                "demoted_ambiguous_rows": sum(on_threshold),
                "divider_settled": divider_settled,
                "displayed_only_demotions": displayed_only if divider_settled else [],
            })
    return records, precision


def guard_demoted_bound(guard_grid_records: list[dict]) -> dict[str, int]:
    """The demoted count's true range over the whole guard grid, accounting for display-rounding
    ambiguity: the smallest ``demoted_min`` and the largest ``demoted_max`` across every cell
    (see ``build_guard_grid``)."""
    return {
        "min": min(r["demoted_min"] for r in guard_grid_records),
        "max": max(r["demoted_max"] for r in guard_grid_records),
    }


def _minmax(records: list[dict], key: str) -> dict[str, int]:
    values = [record["counts"][key] for record in records]
    return {"min": min(values), "max": max(values)}


def grid_minmax_summary(records: list[dict]) -> dict[str, dict[str, int]]:
    return {key: _minmax(records, key) for key in _COUNT_KEYS}


def _regime_b_margin(counts: dict[str, int]) -> int:
    return counts["regime_b"] - max(counts["recovered"], counts["regime_a"], counts["other"])


_REGIME_COMBO_KEYS = ("recovered", "regime_a_marginal", "regime_b_marginal")
_GUARD_COMBO_KEYS = ("vacuous_set_frac", "vacuous_overcover_frac", "overcover_marginal")


def _combo_fields(record: dict) -> dict:
    if all(k in record for k in _REGIME_COMBO_KEYS):
        return {k: record[k] for k in _REGIME_COMBO_KEYS}
    if all(k in record for k in _GUARD_COMBO_KEYS):
        return {k: record[k] for k in _GUARD_COMBO_KEYS}
    raise ValueError(f"record has neither a regime nor a guard threshold combo: {record!r}")


def regime_b_dominance(all_records: list[dict]) -> dict:
    """Whether regime B is the strictly largest of the four labels in every grid combination,
    computed by loop over both grids together, plus the single combination with the smallest
    margin (the one closest to breaking it)."""
    ranked = sorted(all_records, key=lambda record: _regime_b_margin(record["counts"]))
    closest = ranked[0]
    closest_margin = _regime_b_margin(closest["counts"])
    return {
        "always_largest": closest_margin > 0,
        "closest_margin": closest_margin,
        "closest_combo": _combo_fields(closest),
        "closest_counts": closest["counts"],
    }


_COUNT_NOUNS: dict[str, str] = {
    "non_demoted": "non demoted rows",
    "recovered": "recovered rows",
    "regime_a": "regime A rows",
    "regime_b": "regime B rows",
    "other": "other rows",
    "demoted": "demoted rows",
    "demoted_recovered": "demoted rows that would meet the recovered threshold",
    "demoted_regime_a": "demoted rows that would meet the regime A threshold",
}


def _sentence_default_counts(counts: dict[str, int], expected: dict[str, int] | None = None) -> str:
    expected = EXPECTED_DEFAULT_COUNTS if expected is None else expected
    head = (
        f"The headline leaderboard has {counts['non_demoted']} non demoted rows. "
        f"Of those, {counts['recovered']} are recovered, {counts['regime_a']} are regime A, "
        f"{counts['regime_b']} are regime B, and {counts['other']} are other. "
        f"There are {counts['demoted']} demoted rows, of which {counts['demoted_recovered']} "
        f"would meet the recovered threshold and {counts['demoted_regime_a']} would meet the "
        f"regime A threshold before demotion. "
    )
    moved = [key for key, want in expected.items() if counts[key] != want]
    if not moved:
        return head + ("These eight numbers reproduce the counts reported "
                       "in the paper's two regimes section exactly.")
    listed = ", ".join(f"{counts[key]} {_COUNT_NOUNS[key]} where the reviewed version reported "
                       f"{expected[key]}" for key in moved)
    return head + ("The board changed after the review of the paper, so these are not all the "
                   f"counts of the reviewed version: {listed}.")


def _render_regime_table(records: list[dict]) -> list[str]:
    lines = [
        "| recovered | regime A marginal | regime B marginal | non demoted | recovered | "
        "regime A | regime B | other | demoted | rows relabelled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        c = r["counts"]
        lines.append(
            f"| {r['recovered']:.2f} | {r['regime_a_marginal']:.2f} | {r['regime_b_marginal']:.2f} | "
            f"{c['non_demoted']} | {c['recovered']} | {c['regime_a']} | {c['regime_b']} | "
            f"{c['other']} | {c['demoted']} | {r['rows_relabelled']} |"
        )
    return lines


def _render_guard_table(records: list[dict]) -> list[str]:
    lines = [
        "| set/K demotion | over cover set/K | over cover marginal | non demoted | recovered | "
        "regime A | regime B | other | demoted | demoted min | demoted max | ambiguous rows | "
        "rows relabelled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        c = r["counts"]
        lines.append(
            f"| {r['vacuous_set_frac']:.2f} | {r['vacuous_overcover_frac']:.2f} | "
            f"{r['overcover_marginal']:.2f} | {c['non_demoted']} | {c['recovered']} | "
            f"{c['regime_a']} | {c['regime_b']} | {c['other']} | {c['demoted']} | "
            f"{r['demoted_min']} | {r['demoted_max']} | {r['demoted_ambiguous_rows']} | "
            f"{r['rows_relabelled']} |"
        )
    return lines


_COUNT_LABELS = {
    "non_demoted": "non demoted",
    "recovered": "recovered",
    "regime_a": "regime A",
    "regime_b": "regime B",
    "other": "other",
    "demoted": "demoted",
}

_COMBO_LABELS = {
    "recovered": "recovered",
    "regime_a_marginal": "regime A marginal",
    "regime_b_marginal": "regime B marginal",
    "vacuous_set_frac": "set/K demotion",
    "vacuous_overcover_frac": "over cover set/K",
    "overcover_marginal": "over cover marginal",
}


def _join_with_and(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _range_clause(key: str, vals: dict[str, int], qualifier: str | None = None) -> str:
    label = _COUNT_LABELS.get(key, key.replace("_", " "))
    if vals["min"] == vals["max"]:
        clause = f"{label} is {vals['min']} in every combination"
    else:
        clause = f"{label} ranges from {vals['min']} to {vals['max']}"
    if qualifier:
        clause += f" {qualifier}"
    return clause


def _demoted_clause_with_bound(minmax_demoted: dict[str, int], bound: dict[str, int]) -> str:
    displayed = _range_clause("demoted", minmax_demoted)
    if bound["min"] == minmax_demoted["min"] and bound["max"] == minmax_demoted["max"]:
        return displayed
    return (
        f"{displayed} at the precision leaderboard.md displays, or as low as {bound['min']} "
        "once every row sitting exactly on a threshold is allowed to fall on the other side "
        "of it instead"
    )


def _sentence_minmax(
    title: str,
    minmax: dict[str, dict[str, int]],
    demoted_bound: dict[str, int] | None = None,
    displayed_precision_keys: frozenset[str] = frozenset(),
) -> str:
    parts = []
    for key, vals in minmax.items():
        if key == "demoted" and demoted_bound is not None:
            parts.append(_demoted_clause_with_bound(vals, demoted_bound))
        elif key in displayed_precision_keys:
            parts.append(_range_clause(key, vals, qualifier="at displayed precision"))
        else:
            parts.append(_range_clause(key, vals))
    return f"Over the {title}, " + _join_with_and(parts) + "."


def _sentence_guard_ambiguity(guard_grid_records: list[dict], precision: str) -> str:
    if precision != "displayed":
        return (
            "This guard grid used the unrounded row JSONs under --outputs-dir, so every count "
            "above came from each row's own unrounded marginal, mean set size, and number of "
            "classes rather than from the displayed table, and carries no rounding ambiguity."
        )
    cells_with_ambiguity = sum(1 for r in guard_grid_records if r["demoted_ambiguous_rows"] > 0)
    max_ambiguous = max(r["demoted_ambiguous_rows"] for r in guard_grid_records)
    return (
        f"{cells_with_ambiguity} of the {len(guard_grid_records)} guard cells above have at "
        f"least one row sitting exactly on a threshold, with as many as {max_ambiguous} such "
        "rows in a single cell. The demoted column of the guard threshold grid table gives the "
        "count at displayed precision for every cell, the demoted min and demoted max columns give "
        "its lower and upper bounds, and the ambiguous rows column gives the number of rows sitting "
        "on a threshold."
    )


def _sentence_dominance(dominance: dict) -> str:
    combo = dominance["closest_combo"]
    counts = dominance["closest_counts"]
    combo_parts = [
        f"{_COMBO_LABELS.get(key, key.replace('_', ' '))} {value:.2f}"
        for key, value in combo.items()
    ]
    combo_desc = _join_with_and(combo_parts)
    next_largest = max(counts["recovered"], counts["regime_a"], counts["other"])
    if dominance["always_largest"]:
        verdict = "Regime B remains the largest of the four labels in every combination tested."
    else:
        verdict = "Regime B is not the largest of the four labels in every combination tested."
    return (
        f"{verdict} The combination closest to breaking this is {combo_desc}, where regime B has "
        f"{counts['regime_b']} rows against a next largest count of {next_largest} rows, a margin "
        f"of {dominance['closest_margin']} rows."
    )


_GUARD_PRECISION_PARAGRAPH_DISPLAYED = (
    "leaderboard.md prints set/K to two decimal places and marginal to three decimal "
    "places, and the guard grid below recomputes each row's demotion from those displayed "
    "values at two and three decimals rather than from the unrounded values a row was "
    "actually computed from. The paper's own headline demotion, used for the default "
    "counts above and for the regime threshold grid, carries no such risk: it is read "
    "here straight from the table's own divider line, computed once from the unrounded "
    "values when leaderboard.md was generated. A row whose displayed set/K or marginal "
    "lands exactly on one of a guard cell's thresholds could have an unrounded value on "
    "either side of that threshold, so that cell's demoted count cannot be settled from "
    "the displayed table alone. Each guard row below therefore states the demoted count "
    "at displayed precision, the minimum it could take if every such row fell on the "
    "other side of its threshold instead, the maximum, which equals the displayed count "
    "because the table's own convention counts a tie as meeting the threshold, and how "
    "many rows were on a threshold for that cell. The exact grid, computed from every "
    "row's own unrounded marginal, mean set size, and number of classes, is produced "
    "with --outputs-dir where every headline row has its unrounded file."
)

_GUARD_PRECISION_PARAGRAPH_UNROUNDED = (
    "leaderboard.md prints set/K to two decimal places and marginal to three decimal "
    "places, but this guard grid was computed with --outputs-dir, so it recomputes each "
    "row's demotion from the unrounded per-run values under --outputs-dir rather than "
    "from those displayed values. Every row's true marginal and set/K are therefore "
    "known exactly here, so the ambiguous rows column below is zero for every guard "
    "cell and the demoted minimum equals the demoted maximum throughout."
)


def _guard_precision_paragraph(precision: str) -> str:
    if precision == "displayed":
        return _GUARD_PRECISION_PARAGRAPH_DISPLAYED
    return _GUARD_PRECISION_PARAGRAPH_UNROUNDED


def _guard_default_cell_note(guard_grid_records: list[dict]) -> str | None:
    for record in guard_grid_records:
        if record.get("divider_settled"):
            note = (
                "The one exception is the guard cell whose three thresholds equal the "
                "defaults: leaderboard.md's own divider was computed with exactly those "
                "thresholds, so every row's demoted status there is already settled by "
                "the file's divider rather than merely bounded, and that cell's demoted "
                "minimum equals its demoted maximum even though some of its rows sit "
                "exactly on a threshold at displayed precision."
            )
            displayed_only = record.get("displayed_only_demotions", [])
            if displayed_only:
                n = len(displayed_only)
                rows_word = "row" if n == 1 else "rows"
                verb = "displays" if n == 1 else "display"
                falls = "falls" if n == 1 else "fall"
                note += (
                    f" The counts of that cell are the divider's. The displayed values alone "
                    f"would demote {n} more {rows_word} there ({', '.join(displayed_only)}), "
                    f"which {verb} a value exactly on a threshold while the unrounded value "
                    f"the divider used {falls} short of it. The displayed table cannot confirm "
                    f"that; the grid computed with --outputs-dir does."
                )
            return note
    return None


def render_markdown(payload: dict) -> str:
    precision = payload["precision"]
    lines = [
        "# Regime and guard threshold sensitivity",
        "",
        "This file is generated by `scripts/threshold_sensitivity.py` from `leaderboard.md`. "
        "Do not hand edit it.",
        "",
        "## Default counts",
        "",
        _sentence_default_counts(payload["default_counts"], payload.get("expected_default_counts")),
        "",
        "## Regime threshold grid",
        "",
        "The regime label tests each row in a fixed order, recovered first, then regime A, "
        "then regime B, then other. A row only reaches the regime B test once it has failed "
        "both the recovered test and the regime A test, so a row whose marginal sits between "
        "the regime A marginal threshold and the regime B marginal threshold is labelled "
        "regime A whenever the regime A threshold is not above the regime B threshold, never "
        "regime B. This is why some regime B counts in the table below do not move when the "
        "regime B marginal threshold rises past the regime A marginal threshold, because "
        "regime A already claimed those rows first.",
        "",
        "Each row varies the recovered threshold, the regime A marginal threshold, and the "
        "regime B marginal threshold used by the regime label, while the demotion guard stays "
        "at its default values.",
        "",
    ]
    lines += _render_regime_table(payload["regime_grid"])
    lines += [
        "",
        "## Guard threshold grid",
        "",
        _guard_precision_paragraph(precision),
        "",
    ]
    default_cell_note = _guard_default_cell_note(payload["guard_grid"])
    if default_cell_note is not None:
        lines.append(default_cell_note)
        lines.append("")
    lines += [
        "Each row varies the demotion guard thresholds, while the regime label thresholds stay "
        "at their default values.",
        "",
    ]
    lines += _render_guard_table(payload["guard_grid"])
    lines += ["", "## Summary", ""]
    lines.append(_sentence_minmax("regime threshold grid", payload["summary"]["regime_grid"]))
    lines.append("")
    lines.append(_sentence_minmax(
        "guard threshold grid",
        payload["summary"]["guard_grid"],
        demoted_bound=payload["summary"]["guard_grid_demoted_bound"],
        displayed_precision_keys=(
            frozenset({"non_demoted", "recovered"}) if precision == "displayed" else frozenset()
        ),
    ))
    lines.append("")
    lines.append(_sentence_guard_ambiguity(payload["guard_grid"], precision))
    lines.append("")
    lines.append(_sentence_dominance(payload["summary"]["regime_b_dominance"]))
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threshold_sensitivity",
        description=(
            "Sweep the regime and guard thresholds over the headline leaderboard and report "
            "how the counts move."
        ),
    )
    parser.add_argument("--leaderboard", default="leaderboard.md", help="Path to the leaderboard.md pipe-table.")
    parser.add_argument(
        "--json-out", default="outputs/analysis/threshold_sensitivity.json",
        help="Path to write the JSON report to.",
    )
    parser.add_argument(
        "--md-out", default="paper/camera_ready/THRESHOLD_SENSITIVITY.md",
        help="Path to write the markdown report to.",
    )
    parser.add_argument(
        "--outputs-dir", default=None,
        help=(
            "Directory of unrounded per-row JSONs (as `pcpbench run --out` writes them, named "
            "<dataset>_<method>.json). When given, the guard grid recomputes demotion from "
            "each row's unrounded marginal, mean_set_size, and n_classes instead of the "
            "values leaderboard.md displays, and the report is marked precision: unrounded. "
            "Fails loudly, naming the first missing row, if any headline row lacks a JSON "
            "file here."
        ),
    )
    parser.add_argument(
        "--allow-count-change", action="store_true",
        help=(
            "Proceed when the default counts differ from the counts the reviewed version of "
            "the paper reported, which happens once a rerun has changed the board. The report "
            "then holds both sets of counts. A board with another number of headline rows "
            "still stops."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    leaderboard_path = Path(args.leaderboard)
    text = leaderboard_path.read_text()
    all_rows = al.parse_leaderboard(text)
    rows = al.select_headline(all_rows)

    default_counts = reproduce_default_counts(rows)
    mismatches = {
        key: (default_counts[key], expected)
        for key, expected in EXPECTED_DEFAULT_COUNTS.items()
        if default_counts[key] != expected
    }
    print(f"default counts (from {leaderboard_path}): {default_counts}")
    n_expected = EXPECTED_DEFAULT_COUNTS["non_demoted"] + EXPECTED_DEFAULT_COUNTS["demoted"]
    same_rows = default_counts["non_demoted"] + default_counts["demoted"] == n_expected
    if mismatches and args.allow_count_change and same_rows:
        print(
            "NOTE: the default counts differ from the counts of the reviewed version of the "
            "paper; continuing because --allow-count-change was given:",
            file=sys.stderr,
        )
        for key, (found, expected) in mismatches.items():
            print(f"  {key}: found {found}, reviewed version {expected}", file=sys.stderr)
    elif mismatches:
        print(
            "DISCREPANCY: the default combination does not reproduce the paper's counts; "
            "stopping rather than proceeding on a different subset:",
            file=sys.stderr,
        )
        for key, (found, expected) in mismatches.items():
            print(f"  {key}: found {found}, expected {expected}", file=sys.stderr)
        return 1

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else None

    regime_grid = build_regime_grid(rows)
    try:
        guard_grid, precision = build_guard_grid(rows, outputs_dir=outputs_dir)
    except MissingUnroundedRowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "regime_grid": grid_minmax_summary(regime_grid),
        "guard_grid": grid_minmax_summary(guard_grid),
        "guard_grid_demoted_bound": guard_demoted_bound(guard_grid),
        "regime_b_dominance": regime_b_dominance(regime_grid + guard_grid),
    }

    payload = {
        "leaderboard_path": str(leaderboard_path),
        "outputs_dir": str(outputs_dir) if outputs_dir is not None else None,
        "precision": precision,
        "n_headline_rows": len(rows),
        "default_counts": default_counts,
        "expected_default_counts": EXPECTED_DEFAULT_COUNTS,
        "regime_grid": regime_grid,
        "guard_grid": guard_grid,
        "summary": summary,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {json_out}")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(payload))
    print(f"wrote {md_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
