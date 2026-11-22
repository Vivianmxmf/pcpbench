"""Tests for scripts/threshold_sensitivity.py (a standalone stdlib script, loaded here by file
path, same pattern as tests/test_analyze_leaderboard.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AL_PATH = _REPO_ROOT / "scripts" / "analyze_leaderboard.py"
_TS_PATH = _REPO_ROOT / "scripts" / "threshold_sensitivity.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


al = _load_module("analyze_leaderboard", _AL_PATH)
ts = _load_module("threshold_sensitivity", _TS_PATH)


def test_label_regime_explicit_default_kwargs_match_positional_call():
    cases = [(0.88, 0.56), (0.36, 0.00), (0.95, 0.90), (0.83, 0.60)]
    for marginal, worst_class in cases:
        positional = al.label_regime(marginal, worst_class)
        with_defaults = al.label_regime(
            marginal, worst_class,
            recovered=0.80, regime_a_marginal=0.85, regime_b_marginal=0.80,
        )
        assert positional == with_defaults


_EIGHT_ROW_TABLE = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | split | 0.950 | 0.900 | 0.050 | 3.0 | 0.05 |
| ntu60_xsub | mondrian | 0.900 | 0.820 | 0.100 | 3.0 | 0.05 |
| ntu60_xsub | m3 | 0.700 | 0.830 | 0.100 | 3.0 | 0.05 |
| ntu60_xsub | m4 | 0.820 | 0.830 | 0.100 | 3.0 | 0.05 |
| ntu60_xsub | m5 | 0.880 | 0.560 | 0.400 | 3.0 | 0.05 |
| ntu60_xsub | m6 | 0.360 | 0.000 | 1.000 | 3.0 | 0.05 |
| ntu60_xsub | m7 | 0.830 | 0.600 | 0.300 | 3.0 | 0.05 |
| ntu60_xsub | m8 | 0.990 | 0.790 | 0.050 | 3.0 | 0.05 |
"""


def test_hand_built_eight_rows_default_labels_and_recovered_boundary_move():
    rows = al.parse_leaderboard(_EIGHT_ROW_TABLE)
    assert len(rows) == 8
    assert all(not r.demoted for r in rows)

    default_combo = ts.RegimeCombo(
        ts.DEFAULT_RECOVERED, ts.DEFAULT_REGIME_A_MARGINAL, ts.DEFAULT_REGIME_B_MARGINAL
    )
    raised_combo = ts.RegimeCombo(0.85, ts.DEFAULT_REGIME_A_MARGINAL, ts.DEFAULT_REGIME_B_MARGINAL)

    default_labels = [ts.regime_grid_status(r, default_combo)[1] for r in rows]
    raised_labels = [ts.regime_grid_status(r, raised_combo)[1] for r in rows]

    expected_default = [
        al.RECOVERED, al.RECOVERED, al.RECOVERED, al.RECOVERED,
        al.REGIME_A, al.REGIME_B, al.REGIME_OTHER, al.REGIME_A,
    ]
    assert default_labels == expected_default

    expected_raised = [
        al.RECOVERED, al.REGIME_A, al.REGIME_B, al.REGIME_OTHER,
        al.REGIME_A, al.REGIME_B, al.REGIME_OTHER, al.REGIME_A,
    ]
    assert raised_labels == expected_raised

    moved = {i for i in range(8) if default_labels[i] != raised_labels[i]}
    assert moved == {1, 2, 3}


def test_is_vacuous_row_defaults_match_leaderboard_module_constants():
    assert ts.VACUOUS_SET_FRAC == pytest.approx(0.9)
    assert ts.VACUOUS_OVERCOVER_FRAC == pytest.approx(0.60)
    assert ts.OVERCOVER_MARGINAL == pytest.approx(0.95)

    assert ts.is_vacuous_row(marginal=0.70, set_over_k=0.95) is True
    assert ts.is_vacuous_row(marginal=0.99, set_over_k=0.70) is True
    assert ts.is_vacuous_row(marginal=0.80, set_over_k=0.70) is False
    assert ts.is_vacuous_row(marginal=0.99, set_over_k=0.20) is False
    assert ts.is_vacuous_row(
        marginal=0.80, set_over_k=0.55,
        vacuous_set_frac=0.95, vacuous_overcover_frac=0.50, overcover_marginal=0.75,
    ) is True


def _real_headline_rows():
    real_leaderboard = _REPO_ROOT / "leaderboard.md"
    if not real_leaderboard.exists():
        pytest.skip("leaderboard.md not present in this checkout")
    rows_all = al.parse_leaderboard(real_leaderboard.read_text())
    return al.select_headline(rows_all)


def test_recovered_count_never_increases_as_recovered_rises():
    rows = _real_headline_rows()
    thresholds = sorted(ts.REGIME_GRID_RECOVERED)
    counts = [
        ts.regime_grid_counts(
            rows,
            ts.RegimeCombo(t, ts.DEFAULT_REGIME_A_MARGINAL, ts.DEFAULT_REGIME_B_MARGINAL),
        )["recovered"]
        for t in thresholds
    ]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


def test_demoted_count_never_decreases_as_setk_threshold_falls():
    rows = _real_headline_rows()
    thresholds_desc = sorted(ts.GUARD_GRID_SET_FRAC, reverse=True)
    counts = [
        ts.guard_grid_counts(
            rows,
            ts.GuardCombo(t, ts.VACUOUS_OVERCOVER_FRAC, ts.OVERCOVER_MARGINAL),
        )["demoted"]
        for t in thresholds_desc
    ]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))


def test_a_board_of_another_size_stops_even_when_a_count_change_is_allowed(tmp_path, capsys):
    """The headline has the same rows after any rerun, so a different row count means a different
    subset, which the report must never be computed on."""
    real_leaderboard = _REPO_ROOT / "leaderboard.md"
    if not real_leaderboard.exists():
        pytest.skip("leaderboard.md not present in this checkout")
    row = _real_headline_rows()[0]
    kept = [line for line in real_leaderboard.read_text().splitlines(keepends=True)
            if not line.startswith(f"| {row.dataset} | {row.method} |")]
    board = tmp_path / "leaderboard.md"
    board.write_text("".join(kept))
    json_out = tmp_path / "out.json"
    rc = ts.main(["--leaderboard", str(board), "--json-out", str(json_out),
                  "--md-out", str(tmp_path / "out.md"), "--allow-count-change"])
    assert rc == 1
    assert not json_out.exists()
    assert "different subset" in capsys.readouterr().err


def test_main_runs_on_the_committed_board_when_a_count_change_is_allowed(tmp_path):
    """Whatever rerun the committed board holds, the report is computed on the same headline rows
    and keeps the counts of the reviewed version beside its own."""
    real_leaderboard = _REPO_ROOT / "leaderboard.md"
    if not real_leaderboard.exists():
        pytest.skip("leaderboard.md not present in this checkout")
    json_out = tmp_path / "out.json"
    rc = ts.main(["--leaderboard", str(real_leaderboard), "--json-out", str(json_out),
                  "--md-out", str(tmp_path / "out.md"), "--allow-count-change"])
    assert rc == 0
    payload = json.loads(json_out.read_text())
    expected = ts.EXPECTED_DEFAULT_COUNTS
    assert payload["expected_default_counts"] == expected
    counts = payload["default_counts"]
    assert counts["non_demoted"] + counts["demoted"] == expected["non_demoted"] + expected["demoted"]
    assert counts["recovered"] + counts["regime_a"] + counts["regime_b"] + counts["other"] == counts["non_demoted"]


_DEFAULT_GUARD_KEY = (
    ts.VACUOUS_SET_FRAC, ts.VACUOUS_OVERCOVER_FRAC, ts.OVERCOVER_MARGINAL,
)


def _guard_record(guard_grid: list[dict], key: tuple[float, float, float]) -> dict:
    for record in guard_grid:
        if (record["vacuous_set_frac"], record["vacuous_overcover_frac"], record["overcover_marginal"]) == key:
            return record
    raise AssertionError(f"no guard record for {key}")


def test_every_record_counts_sum_to_the_headline_row_count():
    rows = _real_headline_rows()
    regime_grid = ts.build_regime_grid(rows)
    guard_grid, _precision = ts.build_guard_grid(rows)
    for record in regime_grid + guard_grid:
        c = record["counts"]
        assert c["recovered"] + c["regime_a"] + c["regime_b"] + c["other"] == c["non_demoted"]
        assert c["non_demoted"] + c["demoted"] == len(rows)


def test_dominance_margin_is_recomputed_by_loop_and_matches_the_summary():
    rows = _real_headline_rows()
    regime_grid = ts.build_regime_grid(rows)
    guard_grid, _precision = ts.build_guard_grid(rows)
    all_records = regime_grid + guard_grid

    margins = []
    for record in all_records:
        c = record["counts"]
        margins.append(c["regime_b"] - max(c["recovered"], c["regime_a"], c["other"]))
    loop_min_margin = min(margins)

    dominance = ts.regime_b_dominance(all_records)
    assert dominance["closest_margin"] == loop_min_margin
    assert dominance["always_largest"] == (loop_min_margin > 0)


_TWO_ROW_TABLE_OFF_EVERY_THRESHOLD = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | split | 0.500 | 0.500 | 0.500 | 3.0 | 0.50 |
| ntu60_xsub | mondrian | 0.993 | 0.700 | 0.100 | 4.2 | 0.70 |
"""

_TWO_ROW_TABLE_ONE_ON_A_THRESHOLD = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | split | 0.700 | 0.700 | 0.100 | 5.4 | 0.90 |
| ntu60_xsub | mondrian | 0.500 | 0.500 | 0.500 | 3.0 | 0.50 |
"""


def test_ambiguous_rows_zero_off_threshold_positive_on_threshold():
    rows_off = al.select_headline(al.parse_leaderboard(_TWO_ROW_TABLE_OFF_EVERY_THRESHOLD))
    assert len(rows_off) == 2
    grid_off, _precision = ts.build_guard_grid(rows_off)
    record_off = _guard_record(grid_off, _DEFAULT_GUARD_KEY)
    assert record_off["demoted_ambiguous_rows"] == 0
    assert record_off["demoted_min"] == record_off["demoted_max"] == record_off["counts"]["demoted"]

    rows_on = al.select_headline(al.parse_leaderboard(_TWO_ROW_TABLE_ONE_ON_A_THRESHOLD))
    assert len(rows_on) == 2
    grid_on, _precision = ts.build_guard_grid(rows_on)
    record_on = _guard_record(grid_on, (ts.VACUOUS_SET_FRAC, 0.50, ts.OVERCOVER_MARGINAL))
    assert record_on["demoted_ambiguous_rows"] > 0
    assert record_on["demoted_min"] < record_on["demoted_max"]
    assert not record_on["divider_settled"]
    assert record_on["displayed_only_demotions"] == []


_TABLE_WITH_A_DIVIDER_AND_A_ROW_ON_A_THRESHOLD = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | acp | 0.950 | 0.900 | 0.000 | 39.4 | 0.66 |
| ntu60_xsub | split | 0.500 | 0.500 | 0.500 | 3.0 | 0.50 |
| *(vacuous below)* | | | | | | |
| ntu60_xsub | weighted | 0.990 | 0.950 | 0.000 | 57.0 | 0.95 |
"""


def test_the_divider_settles_the_default_cell_against_a_row_displayed_on_a_threshold():
    """A row above the divider that displays marginal 0.950 with set/K 0.66: its unrounded
    marginal is below 0.95, or the divider would have demoted it."""
    rows = al.select_headline(al.parse_leaderboard(_TABLE_WITH_A_DIVIDER_AND_A_ROW_ON_A_THRESHOLD))
    assert [row.demoted for row in rows] == [False, False, True]
    assert ts.is_vacuous_row(rows[0].marginal, rows[0].set_over_k)
    grid, precision = ts.build_guard_grid(rows)
    assert precision == "displayed"
    record = _guard_record(grid, _DEFAULT_GUARD_KEY)
    assert record["divider_settled"]
    assert record["displayed_only_demotions"] == ["ntu60_xsub/acp"]
    assert record["rows_relabelled"] == 0
    by_divider = ts.reproduce_default_counts(rows)
    assert record["counts"] == {key: by_divider[key] for key in ts._COUNT_KEYS}
    assert record["demoted_min"] == record["demoted_max"] == 1
    assert record["demoted_ambiguous_rows"] == 1

    neighbour = _guard_record(grid, (0.85, ts.VACUOUS_OVERCOVER_FRAC, ts.OVERCOVER_MARGINAL))
    assert not neighbour["divider_settled"]
    assert neighbour["displayed_only_demotions"] == []
    assert (neighbour["demoted_min"], neighbour["demoted_max"]) == (1, 2)

    md = ts.render_markdown(_payload_for_markdown(rows))
    assert "is already settled by the file's divider" in md
    assert ("The counts of that cell are the divider's. The displayed values alone would "
            "demote 1 more row there (ntu60_xsub/acp), which displays a value exactly on a "
            "threshold while the unrounded value the divider used falls short of it. The "
            "displayed table cannot confirm that; the grid computed with --outputs-dir does.") in md


_TABLE_WITH_A_ROW_ON_ONE_THRESHOLD_AND_PAST_ANOTHER = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | acp | 0.950 | 0.900 | 0.000 | 57.0 | 0.95 |
| ntu60_xsub | split | 0.500 | 0.500 | 0.500 | 3.0 | 0.50 |
| *(vacuous below)* | | | | | | |
| ntu60_xsub | weighted | 0.990 | 0.950 | 0.000 | 58.2 | 0.97 |
"""


def test_a_tie_that_does_not_decide_the_demotion_does_not_excuse_the_divider():
    """The first row displays marginal 0.950, a tie, but its set/K of 0.95 is past the 0.90 line
    whatever the rounding, so a divider that keeps it was not computed with the default
    thresholds."""
    rows = al.select_headline(al.parse_leaderboard(_TABLE_WITH_A_ROW_ON_ONE_THRESHOLD_AND_PAST_ANOTHER))
    assert [row.demoted for row in rows] == [False, False, True]
    assert ts.is_vacuous_row_min(rows[0].marginal, rows[0].set_over_k)
    grid, _precision = ts.build_guard_grid(rows)
    record = _guard_record(grid, _DEFAULT_GUARD_KEY)
    assert not record["divider_settled"]
    assert record["displayed_only_demotions"] == []
    assert record["rows_relabelled"] == 1
    assert record["counts"]["demoted"] == 2
    assert record["demoted_ambiguous_rows"] == 1
    assert "The counts of that cell are the divider's" not in ts.render_markdown(_payload_for_markdown(rows))


def test_a_disagreement_off_every_threshold_leaves_the_default_cell_as_recomputed():
    """A table without a divider whose second row the guard demotes on values far from any
    threshold: its divider was not computed with the default thresholds, so the default cell is
    not read from it."""
    rows = al.select_headline(al.parse_leaderboard(_TWO_ROW_TABLE_OFF_EVERY_THRESHOLD))
    assert not any(row.demoted for row in rows)
    grid, _precision = ts.build_guard_grid(rows)
    record = _guard_record(grid, _DEFAULT_GUARD_KEY)
    assert not record["divider_settled"]
    assert record["displayed_only_demotions"] == []
    assert record["rows_relabelled"] == 1
    assert record["counts"]["demoted"] == 1


def test_outputs_dir_marks_unrounded_and_fails_loudly_on_missing_json(tmp_path):
    rows = al.select_headline(al.parse_leaderboard(_TWO_ROW_TABLE_OFF_EVERY_THRESHOLD))
    assert len(rows) == 2

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    for row in rows:
        (outputs_dir / f"{row.dataset}_{row.method}.json").write_text(json.dumps({
            "marginal": row.marginal,
            "mean_set_size": row.mean_set_size,
            "n_classes": 10,
        }))

    grid, precision = ts.build_guard_grid(rows, outputs_dir=outputs_dir)
    assert precision == "unrounded"
    assert all(record["demoted_ambiguous_rows"] == 0 for record in grid)
    assert all(record["demoted_min"] == record["demoted_max"] for record in grid)

    displayed_payload = _payload_for_markdown(rows)
    md_displayed = ts.render_markdown(displayed_payload)
    unrounded_payload = _payload_for_markdown(rows, outputs_dir=outputs_dir)
    md_unrounded = ts.render_markdown(unrounded_payload)

    displayed_only = "cannot be settled from the displayed table alone"
    unrounded_only = (
        "recomputes each row's demotion from the unrounded per-run values under --outputs-dir"
    )
    assert displayed_only in md_displayed
    assert displayed_only not in md_unrounded
    assert unrounded_only in md_unrounded
    assert unrounded_only not in md_displayed

    missing_row = rows[0]
    (outputs_dir / f"{missing_row.dataset}_{missing_row.method}.json").unlink()
    with pytest.raises(ts.MissingUnroundedRowError) as excinfo:
        ts.build_guard_grid(rows, outputs_dir=outputs_dir)
    assert missing_row.dataset in str(excinfo.value)
    assert missing_row.method in str(excinfo.value)


def _payload_for_markdown(rows: list, outputs_dir: Path | None = None) -> dict:
    default_counts = ts.reproduce_default_counts(rows)
    regime_grid = ts.build_regime_grid(rows)
    guard_grid, precision = ts.build_guard_grid(rows, outputs_dir=outputs_dir)
    summary = {
        "regime_grid": ts.grid_minmax_summary(regime_grid),
        "guard_grid": ts.grid_minmax_summary(guard_grid),
        "guard_grid_demoted_bound": ts.guard_demoted_bound(guard_grid),
        "regime_b_dominance": ts.regime_b_dominance(regime_grid + guard_grid),
    }
    return {
        "leaderboard_path": "leaderboard.md",
        "outputs_dir": str(outputs_dir) if outputs_dir is not None else None,
        "precision": precision,
        "n_headline_rows": len(rows),
        "default_counts": default_counts,
        "expected_default_counts": ts.EXPECTED_DEFAULT_COUNTS,
        "regime_grid": regime_grid,
        "guard_grid": guard_grid,
        "summary": summary,
    }


def test_exact_count_wording_is_replaced_by_displayed_precision():
    rows = _real_headline_rows()
    payload = _payload_for_markdown(rows)
    md = ts.render_markdown(payload)
    assert "exact count" not in md
    assert "gives the count at displayed precision for every cell" in md
    assert "give its lower and upper bounds" in md
    assert "gives the number of rows sitting on a threshold" in md
