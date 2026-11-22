"""Leaderboard aggregator: collects per-method-per-dataset metric row JSONs (as written by
``pcpbench run --out``) and renders a markdown leaderboard table sorted by worst-class coverage
(best-covered worst class first)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_COLUMNS = (
    "dataset", "method", "marginal", "worst_class", "frac_undercovered",
    "frac_sig_undercovered", "mean_set_size",
)
_OPTIONAL_COLUMNS = frozenset({"frac_sig_undercovered"})

VACUOUS_SET_FRAC = 0.9
VACUOUS_OVERCOVER_FRAC = 0.60
OVERCOVER_MARGINAL = 0.95


def _set_fraction(row: dict) -> float | None:
    n_classes = row.get("n_classes")
    if not isinstance(n_classes, (int, float)) or n_classes <= 0:
        return None
    return row["mean_set_size"] / n_classes


def _is_vacuous(row: dict) -> bool:
    frac = _set_fraction(row)
    if frac is None:
        return False
    return frac >= VACUOUS_SET_FRAC or (
        frac >= VACUOUS_OVERCOVER_FRAC and row["marginal"] >= OVERCOVER_MARGINAL
    )


def _is_invalid(row: dict) -> bool:
    w = row.get("worst_class")
    return not isinstance(w, (int, float)) or not math.isfinite(w)


def aggregate(rows: list[dict]) -> list[dict]:
    """Return ``rows`` sorted by ``worst_class`` descending (best worst-class coverage first),
    with vacuous rows (see ``_is_vacuous``) demoted below every non-vacuous row, and INVALID rows
    (non-finite ``worst_class``, see ``_is_invalid``) demoted below everything -- a NaN cell must
    never outrank a real method."""
    invalid = [f"{r.get('dataset')}/{r.get('method')}" for r in rows if _is_invalid(r)]
    if invalid:
        print(
            "WARNING: leaderboard has "
            f"{len(invalid)} row(s) with non-finite worst_class (demoted to the "
            f"bottom, not ranked): {', '.join(invalid)}",
            file=sys.stderr,
        )
    return sorted(
        rows,
        key=lambda row: (
            1 if _is_invalid(row) else 0,
            1 if _is_vacuous(row) else 0,
            -row["worst_class"] if not _is_invalid(row) else 0.0,
        ),
    )


def to_markdown(rows: list[dict]) -> str:
    """Render ``rows`` as a markdown table with columns ``_COLUMNS`` plus a trailing ``set/K``
    column (``_set_fraction`` formatted to 2 decimals, or ``"?"`` when unknown)."""
    all_columns = _COLUMNS + ("set/K",)
    header = "| " + " | ".join(all_columns) + " |"
    sep = "| " + " | ".join("---" for _ in all_columns) + " |"
    lines = [header, sep]

    vacuous_flags = [_is_vacuous(row) for row in rows]
    insert_at = None
    if any(vacuous_flags) and not all(vacuous_flags) and rows:
        insert_at = vacuous_flags.index(True)

    for i, row in enumerate(rows):
        if insert_at == i:
            lines.append("| " + "*(vacuous below)*" + " |" * len(all_columns))
        cells = []
        for col in _COLUMNS:
            if col in _OPTIONAL_COLUMNS and col not in row:
                cells.append("?")
                continue
            value = row[col]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f"{float(value):.3f}")
            else:
                cells.append(str(value))
        frac = _set_fraction(row)
        cells.append(f"{frac:.2f}" if frac is not None else "?")
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines) + "\n"
    table += (
        "\n_set/K = mean set size / number of classes; rows are demoted "
        f"below the divider when set/K >= {VACUOUS_SET_FRAC:.2f} (near-full "
        f"sets), or when set/K >= {VACUOUS_OVERCOVER_FRAC:.2f} and marginal "
        f">= {OVERCOVER_MARGINAL:.2f} (over-covering with large sets)._\n"
    )
    return table


_REQUIRED_KEYS = ("dataset", "method", "marginal", "worst_class", "frac_undercovered", "mean_set_size")


def datasets_of_board(markdown: str) -> list[str]:
    """Return the datasets a leaderboard markdown lists, sorted and unique."""
    table_rows = [line for line in markdown.splitlines() if line.startswith("|")]
    names: set[str] = set()
    for line in table_rows[2:]:
        cells = line.split("|")
        name = cells[1].strip() if len(cells) > 1 else ""
        if name and not name.startswith("*"):
            names.add(name)
    return sorted(names)


def _load_rows(in_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(in_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skipped {path}: {exc}")
            continue
        if not isinstance(row, dict):
            print(f"skipped {path}: not a JSON object")
            continue
        missing = [key for key in _REQUIRED_KEYS if key not in row]
        if missing:
            print(f"skipped {path}: missing keys {missing}")
            continue
        rows.append(row)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcpbench.leaderboard",
        description="Aggregate pcpbench run row JSONs into a markdown leaderboard.",
    )
    parser.add_argument("--in", dest="in_dir", required=True, help="Directory of row JSON files.")
    parser.add_argument("--out", dest="out_path", required=True, help="Markdown output path.")
    parser.add_argument(
        "--datasets-from", dest="datasets_from", default=None,
        help="A leaderboard markdown. Only rows of the datasets it lists are kept.",
    )
    return parser


def _keep_listed_datasets(rows: list[dict], reference: Path) -> list[dict] | None:
    try:
        keep = set(datasets_of_board(reference.read_text()))
    except OSError as exc:
        print(f"leaderboard: cannot read {reference}: {exc}", file=sys.stderr)
        return None
    if not keep:
        print(f"leaderboard: {reference} lists no dataset, so nothing was written", file=sys.stderr)
        return None
    kept = [row for row in rows if row["dataset"] in keep]
    left_out = sorted({str(row["dataset"]) for row in rows if row["dataset"] not in keep})
    absent = sorted(keep - {row["dataset"] for row in kept})
    print(
        f"leaderboard: kept {len(kept)} rows of the {len(keep) - len(absent)} datasets that "
        f"{reference} lists and that have rows; left out {len(rows) - len(kept)} rows of "
        f"{len(left_out)} other datasets" + (f": {', '.join(left_out)}" if left_out else "")
    )
    if absent:
        print(f"leaderboard: {len(absent)} datasets of {reference} have no row JSON: {', '.join(absent)}")
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    rows = _load_rows(in_dir)
    if args.datasets_from is not None:
        kept = _keep_listed_datasets(rows, Path(args.datasets_from))
        if kept is None:
            return 2
        rows = kept
    ranked = aggregate(rows)
    table = to_markdown(ranked)

    out_path = Path(args.out_path)
    out_path.write_text(table)

    if not ranked:
        print(f"leaderboard: 0 rows -> {out_path}")
        return 0

    finite = [row for row in ranked if not _is_invalid(row)]
    n_invalid = len(ranked) - len(finite)
    invalid_note = f"; {n_invalid} invalid row(s) demoted" if n_invalid else ""

    if not finite:
        print(f"leaderboard: {len(ranked)} rows -> {out_path}; "
              f"WARNING: no valid rows (all {len(ranked)} have non-finite "
              f"worst_class)")
    elif all(_is_vacuous(row) for row in finite):
        best = finite[0]
        print(f"leaderboard: {len(ranked)} rows -> {out_path}; "
              f"WARNING: all valid rows are vacuous (set/K >= {VACUOUS_SET_FRAC:.2f}, "
              f"or set/K >= {VACUOUS_OVERCOVER_FRAC:.2f} with marginal >= "
              f"{OVERCOVER_MARGINAL:.2f}); "
              f"best listed worst_class={best['worst_class']:.3f} (VACUOUS)"
              + invalid_note)
    else:
        best = finite[0]
        frac = _set_fraction(best)
        suffix = f" set/K={frac:.2f}" if frac is not None else ""
        print(f"leaderboard: {len(ranked)} rows -> {out_path}; "
              f"best worst_class={best['worst_class']:.3f} "
              f"({best['dataset']}/{best['method']})" + suffix + invalid_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
