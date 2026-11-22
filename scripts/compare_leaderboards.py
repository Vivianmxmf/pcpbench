#!/usr/bin/env python3
"""Cell-level comparison of two leaderboard markdown files."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

_ANALYZE_PATH = Path(__file__).resolve().parent / "analyze_leaderboard.py"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("analyze_leaderboard_for_compare", _ANALYZE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


al = _load_analyze()

_KEY_COLUMNS = ("dataset", "method")


def _printed_cells(text: str) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    header: list[str] = []
    cells_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not header:
            header = cells
            continue
        if al._is_separator_row(cells) or al._is_divider_row(cells):
            continue
        if len(cells) < len(header):
            continue
        named = dict(zip(header, cells))
        cells_by_key.setdefault((named["dataset"], named["method"]), named)
    return header, cells_by_key


def _rows_by_key(text: str) -> dict[tuple[str, str], "al.Row"]:
    rows: dict[tuple[str, str], al.Row] = {}
    for row in al.parse_leaderboard(text):
        key = (row.dataset, row.method)
        if key in rows:
            raise ValueError(f"duplicate leaderboard row {key[0]}/{key[1]}")
        rows[key] = row
    return rows


def _abs_diff(old: str, new: str) -> float | None:
    try:
        a, b = float(old), float(new)
    except ValueError:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return abs(a - b)


def _key_dict(key: tuple[str, str]) -> dict[str, str]:
    return {"dataset": key[0], "method": key[1]}


def compare_boards(old_text: str, new_text: str) -> dict:
    """Compare two leaderboard texts and return the diff payload."""
    old_rows, new_rows = _rows_by_key(old_text), _rows_by_key(new_text)
    old_header, old_cells = _printed_cells(old_text)
    new_header, new_cells = _printed_cells(new_text)
    compared = [c for c in old_header if c in new_header and c not in _KEY_COLUMNS]

    shared = [k for k in old_rows if k in new_rows]
    changed_rows: list[dict] = []
    per_column: dict[str, int] = {}
    demotion_changes: list[dict] = []
    for key in shared:
        old_row, new_row = old_rows[key], new_rows[key]
        if old_row.demoted != new_row.demoted:
            demotion_changes.append({**_key_dict(key), "demoted_old": old_row.demoted,
                                     "demoted_new": new_row.demoted})
        cells: dict[str, dict] = {}
        for column in compared:
            old_s, new_s = old_cells[key][column], new_cells[key][column]
            if old_s != new_s:
                cells[column] = {"old": old_s, "new": new_s, "abs_diff": _abs_diff(old_s, new_s)}
                per_column[column] = per_column.get(column, 0) + 1
        if not cells:
            continue
        regime_old = al.label_regime(old_row.marginal, old_row.worst_class)
        regime_new = al.label_regime(new_row.marginal, new_row.worst_class)
        changed_rows.append({
            **_key_dict(key),
            "cells": cells,
            "regime_old": regime_old,
            "regime_new": regime_new,
            "regime_changed": regime_old != regime_new,
            "demoted_old": old_row.demoted,
            "demoted_new": new_row.demoted,
        })

    regime_transitions: dict[str, int] = {}
    for row in changed_rows:
        if row["regime_changed"]:
            label = f"{row['regime_old']} -> {row['regime_new']}"
            regime_transitions[label] = regime_transitions.get(label, 0) + 1

    return {
        "columns_compared": compared,
        "columns_only_in_old": [c for c in old_header if c not in new_header],
        "columns_only_in_new": [c for c in new_header if c not in old_header],
        "n_rows_old": len(old_rows),
        "n_rows_new": len(new_rows),
        "n_shared_rows": len(shared),
        "n_changed_rows": len(changed_rows),
        "n_changed_cells": sum(per_column.values()),
        "changed_cells_per_column": {c: per_column[c] for c in compared if c in per_column},
        "rows_only_in_old": [_key_dict(k) for k in old_rows if k not in new_rows],
        "rows_only_in_new": [_key_dict(k) for k in new_rows if k not in old_rows],
        "n_regime_changes": sum(1 for r in changed_rows if r["regime_changed"]),
        "regime_transitions": regime_transitions,
        "demotion_changes": demotion_changes,
        "changed_rows": changed_rows,
    }


def summary_paragraph(payload: dict) -> str:
    """One paragraph of full sentences describing ``payload``."""
    per_column = payload["changed_cells_per_column"]
    if per_column:
        columns = ", ".join(f"{n} in {c}" for c, n in per_column.items())
        cell_sentence = f"The {payload['n_changed_cells']} changed cells fall {columns}."
    else:
        cell_sentence = "No printed cell changed."
    return (
        f"The old board has {payload['n_rows_old']} rows and the new board has "
        f"{payload['n_rows_new']} rows. "
        f"{payload['n_changed_rows']} of the {payload['n_shared_rows']} shared rows changed "
        f"at least one printed cell. {cell_sentence} "
        f"{len(payload['rows_only_in_old'])} rows appear only in the old board and "
        f"{len(payload['rows_only_in_new'])} rows appear only in the new board. "
        f"{payload['n_regime_changes']} changed rows moved to another regime label at the "
        f"default thresholds, and {len(payload['demotion_changes'])} shared rows changed "
        f"their demotion status."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", required=True, type=Path, help="Released leaderboard markdown.")
    parser.add_argument("--new", required=True, type=Path, help="Regenerated leaderboard markdown.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = {"old": str(args.old), "new": str(args.new)}
    payload.update(compare_boards(args.old.read_text(), args.new.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(summary_paragraph(payload))
    print(f"Written to {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
