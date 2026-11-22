import json

import pytest

from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.leaderboard import (
    VACUOUS_SET_FRAC,
    _is_vacuous,
    _set_fraction,
    aggregate,
    datasets_of_board,
    main,
    to_markdown,
)
from pcpbench.runner import run


def _row(dataset, method, worst_class, mean_set_size=2.0, n_classes=5, marginal=0.9):
    return {
        "dataset": dataset,
        "method": method,
        "marginal": marginal,
        "worst_class": worst_class,
        "frac_undercovered": 0.1,
        "mean_set_size": mean_set_size,
        "n_classes": n_classes,
    }


def test_aggregate_sorts_by_worst_class_descending_and_does_not_mutate_input():
    rows = [
        _row("ds1", "split", 0.7),
        _row("ds1", "mondrian", 0.9),
        _row("ds1", "weighted", 0.8),
    ]
    original = [dict(r) for r in rows]

    ranked = aggregate(rows)

    assert [r["worst_class"] for r in ranked] == pytest.approx([0.9, 0.8, 0.7])
    assert [r["method"] for r in ranked] == ["mondrian", "weighted", "split"]
    assert rows == original
    assert rows[0]["worst_class"] == 0.7


def test_to_markdown_contains_header_and_all_entries():
    rows = [
        _row("cifar10c", "split", 0.62),
        _row("ntu60_xsub", "mondrian", 0.81),
    ]
    table = to_markdown(rows)

    assert "| dataset | method | marginal | worst_class | frac_undercovered | frac_sig_undercovered | mean_set_size | set/K |" in table
    for row in rows:
        assert row["dataset"] in table
        assert row["method"] in table
    assert "0.620" in table
    assert "0.810" in table


def test_aggregate_demotes_vacuous_row_below_real_row_despite_higher_worst_class():
    vacuous = _row("ds1", "acp", worst_class=1.0, mean_set_size=4.9, n_classes=5)
    real = _row("ds1", "split", worst_class=0.90, mean_set_size=2.4, n_classes=5)

    ranked = aggregate([vacuous, real])

    assert [r["method"] for r in ranked] == ["split", "acp"]
    assert _is_vacuous(vacuous) is True
    assert _is_vacuous(real) is False


def test_aggregate_preserves_worst_class_order_within_each_group():
    vacuous_low = _row("ds1", "a", worst_class=0.5, mean_set_size=4.9, n_classes=5)
    vacuous_high = _row("ds1", "b", worst_class=0.95, mean_set_size=4.9, n_classes=5)
    real_low = _row("ds1", "c", worst_class=0.6, mean_set_size=2.0, n_classes=5)
    real_high = _row("ds1", "d", worst_class=0.8, mean_set_size=2.0, n_classes=5)

    ranked = aggregate([vacuous_low, real_low, vacuous_high, real_high])

    assert [r["method"] for r in ranked] == ["d", "c", "b", "a"]


def test_is_vacuous_boundary_at_vacuous_set_frac():
    K = 5
    just_below = _row("ds1", "a", worst_class=0.5, mean_set_size=(VACUOUS_SET_FRAC * K) - 0.01, n_classes=K)
    at_or_above = _row("ds1", "b", worst_class=0.5, mean_set_size=VACUOUS_SET_FRAC * K, n_classes=K)

    assert _is_vacuous(just_below) is False
    assert _is_vacuous(at_or_above) is True


def test_is_vacuous_overcover_rule_demotes_large_set_with_high_marginal():
    row = _row("cifar10c_contrast_mild", "acp", worst_class=0.986, mean_set_size=8.7, n_classes=10, marginal=0.997)

    assert _is_vacuous(row) is True


def test_is_vacuous_overcover_rule_keeps_honest_oracle_at_nominal_coverage():
    row = _row("ds1", "oracle", worst_class=0.901, mean_set_size=7.2, n_classes=10, marginal=0.901)

    assert _is_vacuous(row) is False


def test_is_vacuous_near_full_set_demoted_regardless_of_marginal():
    row = _row("ds1", "x", worst_class=0.5, mean_set_size=9.6, n_classes=10, marginal=0.90)

    assert _is_vacuous(row) is True


def test_is_vacuous_large_set_not_demoted_when_marginal_below_overcover_threshold():
    row = _row("ds1", "x", worst_class=0.5, mean_set_size=6.5, n_classes=10, marginal=0.94)

    assert _is_vacuous(row) is False


def test_is_vacuous_set_frac_just_below_overcover_threshold_not_demoted():
    row = _row("ds1", "x", worst_class=0.5, mean_set_size=5.5, n_classes=10, marginal=0.99)

    assert _is_vacuous(row) is False


def test_is_vacuous_missing_n_classes_not_demoted_even_with_high_marginal():
    row_no_k = {
        "dataset": "ds1",
        "method": "x",
        "marginal": 0.99,
        "worst_class": 0.5,
        "frac_undercovered": 0.1,
        "mean_set_size": 9.9,
    }

    assert _set_fraction(row_no_k) is None
    assert _is_vacuous(row_no_k) is False


def test_aggregate_demotes_overcovering_row_below_honest_lower_worst_class_row():
    overcover = _row("cifar10c_contrast_mild", "acp", worst_class=0.986, mean_set_size=8.7, n_classes=10, marginal=0.997)
    honest = _row("ds2", "split", worst_class=0.742, mean_set_size=1.0, n_classes=10, marginal=0.908)

    ranked = aggregate([overcover, honest])

    assert [r["method"] for r in ranked] == ["split", "acp"]
    assert _is_vacuous(honest) is False
    assert _is_vacuous(overcover) is True

    table = to_markdown(ranked)
    assert "*(vacuous below)*" in table
    assert "set/K" in table


def test_set_fraction_none_when_n_classes_missing_and_row_not_demoted():
    row_no_k = {
        "dataset": "legacy_ds",
        "method": "split",
        "marginal": 0.9,
        "worst_class": 0.7,
        "frac_undercovered": 0.1,
        "mean_set_size": 2.0,
    }

    assert _set_fraction(row_no_k) is None
    assert _is_vacuous(row_no_k) is False

    ranked = aggregate([row_no_k])
    assert ranked == [row_no_k]

    table = to_markdown([row_no_k])
    assert "legacy_ds" in table
    lines = [line for line in table.splitlines() if "legacy_ds" in line]
    assert len(lines) == 1
    assert lines[0].rstrip().endswith("| ? |")


def test_to_markdown_inserts_divider_and_legend_for_mixed_rows():
    real = _row("ds1", "split", worst_class=0.9, mean_set_size=2.4, n_classes=5)
    vacuous = _row("ds1", "acp", worst_class=1.0, mean_set_size=4.9, n_classes=5)

    ranked = aggregate([vacuous, real])
    table = to_markdown(ranked)

    assert "set/K" in table
    assert "*(vacuous below)*" in table
    assert "set/K = mean set size / number of classes" in table
    real_idx = table.index("split")
    divider_idx = table.index("*(vacuous below)*")
    vacuous_idx = table.index("acp")
    assert real_idx < divider_idx < vacuous_idx


def test_to_markdown_no_divider_when_all_rows_same_group():
    rows = [
        _row("ds1", "split", worst_class=0.9, mean_set_size=2.4, n_classes=5),
        _row("ds1", "mondrian", worst_class=0.8, mean_set_size=2.0, n_classes=5),
    ]
    table = to_markdown(rows)
    assert "*(vacuous below)*" not in table


def test_main_end_to_end_writes_leaderboard_and_skips_malformed_json(tmp_path, capsys):
    in_dir = tmp_path / "results"
    in_dir.mkdir()
    out_path = tmp_path / "leaderboard.md"

    ds = make_synthetic_shift(K=6, n_per_class=200, shift=0.6, seed=0)
    methods = ["split", "mondrian"]
    for m in methods:
        row = run(ds, m, alpha=0.1, seeds=2)
        assert row["n_classes"] == 6
        (in_dir / f"{m}.json").write_text(json.dumps(row))

    (in_dir / "broken.json").write_text("{not json")

    rc = main(["--in", str(in_dir), "--out", str(out_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "skipped" in captured.out
    assert "broken.json" in captured.out

    table = out_path.read_text()
    assert "| dataset | method | marginal | worst_class | frac_undercovered | frac_sig_undercovered | mean_set_size | set/K |" in table
    for m in methods:
        assert m in table


def test_datasets_of_board_reads_the_dataset_column_and_skips_the_divider():
    ranked = aggregate([_row("d2", "split", 0.5), _row("d1", "oracle", 0.9, mean_set_size=4.9),
                        _row("d2", "oracle", 0.7)])
    board = to_markdown(ranked)
    assert "*(vacuous below)*" in board
    assert datasets_of_board(board) == ["d1", "d2"]
    assert datasets_of_board("") == []


def _write_rows(directory, rows):
    directory.mkdir()
    for row in rows:
        (directory / f"{row['dataset']}_{row['method']}.json").write_text(json.dumps(row))


def test_main_datasets_from_builds_the_board_of_the_listed_datasets_only(tmp_path, capsys):
    """A row JSON of a dataset the reference board lacks never reaches the board."""
    on_board = [_row("d1", "split", 0.5), _row("d1", "oracle", 0.8), _row("d2", "split", 0.6),
                _row("d2", "oracle", 0.9, mean_set_size=4.9)]
    off_board = [_row("d1_alias", "split", 0.55), _row("d1_alias", "oracle", 0.85)]
    _write_rows(tmp_path / "subset", on_board)
    _write_rows(tmp_path / "all", on_board + off_board)
    reference = tmp_path / "reference.md"
    assert main(["--in", str(tmp_path / "subset"), "--out", str(reference)]) == 0
    unfiltered = tmp_path / "unfiltered.md"
    assert main(["--in", str(tmp_path / "all"), "--out", str(unfiltered)]) == 0
    assert "d1_alias" in unfiltered.read_text()
    capsys.readouterr()

    out = tmp_path / "filtered.md"
    rc = main(["--in", str(tmp_path / "all"), "--out", str(out), "--datasets-from", str(reference)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert out.read_text() == reference.read_text()
    assert "left out 2 rows" in printed and "d1_alias" in printed


def test_main_datasets_from_reports_a_listed_dataset_without_rows(tmp_path, capsys):
    _write_rows(tmp_path / "rows", [_row("d1", "split", 0.5)])
    reference = tmp_path / "reference.md"
    reference.write_text(to_markdown([_row("d1", "split", 0.5), _row("d9", "split", 0.4)]))
    out = tmp_path / "out.md"
    assert main(["--in", str(tmp_path / "rows"), "--out", str(out), "--datasets-from", str(reference)]) == 0
    assert "d9" in capsys.readouterr().out
    assert datasets_of_board(out.read_text()) == ["d1"]


@pytest.mark.parametrize("content", [None, "", "no table here\n"])
def test_main_datasets_from_refuses_a_missing_or_empty_reference(tmp_path, capsys, content):
    _write_rows(tmp_path / "rows", [_row("d1", "split", 0.5)])
    reference = tmp_path / "reference.md"
    if content is not None:
        reference.write_text(content)
    out = tmp_path / "out.md"
    rc = main(["--in", str(tmp_path / "rows"), "--out", str(out), "--datasets-from", str(reference)])
    assert rc == 2
    assert not out.exists()
    assert str(reference) in capsys.readouterr().err


def test_main_skips_json_missing_required_keys(tmp_path, capsys):
    in_dir = tmp_path / "results"
    in_dir.mkdir()
    out_path = tmp_path / "leaderboard.md"

    (in_dir / "incomplete.json").write_text(json.dumps({"dataset": "x", "method": "y"}))

    rc = main(["--in", str(in_dir), "--out", str(out_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "skipped" in captured.out
    assert "incomplete.json" in captured.out
    assert out_path.exists()


def test_nonfinite_worst_class_rows_are_demoted_below_all_real_rows(capsys):
    """A row whose worst_class is NaN (e.g. every seed's oracle/acp fit was skipped for an empty
    budget) must never outrank a real row, and must be announced on stderr -- a failed
    regeneration cannot masquerade as a top leaderboard entry."""
    from pcpbench.leaderboard import aggregate

    good = {"dataset": "d", "method": "split", "marginal": 0.9,
            "worst_class": 0.8, "frac_undercovered": 0.1,
            "mean_set_size": 2.0, "n_classes": 10}
    nan_row = {"dataset": "d", "method": "oracle", "marginal": float("nan"),
               "worst_class": float("nan"), "frac_undercovered": float("nan"),
               "mean_set_size": float("nan"), "n_classes": 10}

    ordered = aggregate([nan_row, good])
    assert ordered[0]["method"] == "split"
    assert ordered[-1]["method"] == "oracle"
    assert "non-finite worst_class" in capsys.readouterr().err


def test_all_invalid_leaderboard_reports_no_valid_rows_not_a_nan_best(tmp_path, capsys):
    """When every cell is invalid (non-finite worst_class), main must report 'no valid rows' and
    never print a NaN row as the best result."""
    import json as _json

    from pcpbench.leaderboard import main

    nan_row = {"dataset": "d", "method": "oracle", "marginal": float("nan"),
               "worst_class": float("nan"), "frac_undercovered": float("nan"),
               "mean_set_size": float("nan"), "n_classes": 10,
               "n_valid_seeds": 0, "n_failed_seeds": 10}
    (tmp_path / "d_oracle.json").write_text(_json.dumps(nan_row))
    out = tmp_path / "leaderboard.md"

    rc = main(["--in", str(tmp_path), "--out", str(out)])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "no valid rows" in stdout
    assert "best worst_class" not in stdout


def test_invalid_row_sorts_below_a_vacuous_row(capsys):
    """A NaN row must rank below even a vacuous (demoted) real row, not merely below clean rows."""
    from pcpbench.leaderboard import aggregate

    vacuous = {"dataset": "d", "method": "acp", "marginal": 0.99,
               "worst_class": 1.0, "frac_undercovered": 0.0,
               "mean_set_size": 9.5, "n_classes": 10}
    nan_row = {"dataset": "d", "method": "oracle", "marginal": float("nan"),
               "worst_class": float("nan"), "frac_undercovered": float("nan"),
               "mean_set_size": float("nan"), "n_classes": 10}
    ordered = aggregate([nan_row, vacuous])
    assert ordered[0]["method"] == "acp"
    assert ordered[-1]["method"] == "oracle"


def _data_lines(table):
    return [ln for ln in table.splitlines()
            if ln.startswith("| ") and "---" not in ln and not ln.startswith("| dataset")
            and "vacuous below" not in ln]


def test_header_has_eight_columns_with_sig_column_after_frac_undercovered():
    table = to_markdown([_row("d", "split", 0.8)])
    header = table.splitlines()[0]
    names = [c.strip() for c in header.strip("|").split("|")]
    assert names == ["dataset", "method", "marginal", "worst_class", "frac_undercovered",
                     "frac_sig_undercovered", "mean_set_size", "set/K"]
    sep = table.splitlines()[1]
    assert len(sep.strip("|").split("|")) == 8


def test_row_without_sig_key_renders_question_mark():
    table = to_markdown([_row("d", "split", 0.8)])
    cells = [c.strip() for c in _data_lines(table)[0].strip("|").split("|")]
    assert len(cells) == 8
    assert cells[4] == "0.100"
    assert cells[5] == "?"
    assert cells[6] == "2.000"


def test_row_with_sig_key_renders_the_number():
    row = {**_row("d", "split", 0.8), "frac_sig_undercovered": 0.0833333}
    table = to_markdown([row])
    cells = [c.strip() for c in _data_lines(table)[0].strip("|").split("|")]
    assert cells[5] == "0.083"


def test_vacuous_divider_spans_eight_columns():
    vacuous = _row("d", "acp", worst_class=1.0, mean_set_size=4.9, n_classes=5)
    real = _row("d", "split", worst_class=0.9, mean_set_size=2.4, n_classes=5)
    table = to_markdown(aggregate([vacuous, real]))
    divider = next(ln for ln in table.splitlines() if "vacuous below" in ln)
    assert divider.count("|") == 9


def test_guard_demotes_the_same_rows_with_or_without_the_sig_key():
    fixtures = [
        _row("d", "acp", worst_class=1.0, mean_set_size=4.9, n_classes=5),
        _row("d", "split", worst_class=0.9, mean_set_size=2.4, n_classes=5),
        _row("d", "weighted", worst_class=0.95, mean_set_size=3.2, n_classes=5,
             marginal=0.97),
        _row("d", "mondrian", worst_class=0.7),
        {**_row("d", "oracle", worst_class=float("nan")), "marginal": float("nan")},
    ]
    with_sig = [{**r, "frac_sig_undercovered": 0.05} for r in fixtures]
    order_old = [(r["method"], _is_vacuous(r)) for r in aggregate(fixtures)]
    order_new = [(r["method"], _is_vacuous(r)) for r in aggregate(with_sig)]
    assert order_old == order_new


def test_main_loads_a_row_json_without_the_sig_key(tmp_path):
    in_dir = tmp_path / "results"
    in_dir.mkdir()
    (in_dir / "old.json").write_text(json.dumps(_row("d", "split", 0.8)))
    out = tmp_path / "lb.md"
    assert main(["--in", str(in_dir), "--out", str(out)]) == 0
    cells = [c.strip() for c in _data_lines(out.read_text())[0].strip("|").split("|")]
    assert cells[1] == "split" and cells[5] == "?"


def _load_analyze():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_leaderboard.py"
    spec = importlib.util.spec_from_file_location("analyze_leaderboard_task4a", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_OLD_HEADER_BOARD = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_contrast_s3 | oracle | 0.900 | 0.850 | 0.300 | 1.500 | 0.15 |
| cifar100c_contrast_s3 | split | 0.880 | 0.400 | 0.700 | 3.000 | 0.03 |
| ntu60_xsub | mondrian | 0.700 | 0.100 | 0.900 | 2.000 | 0.03 |
| cifar10c_fog_s5 | weighted | 0.820 | 0.600 | 0.500 | 1.200 | 0.12 |
| *(vacuous below)* | | | | | | | |
| cifar10c_contrast_s3 | acp | 0.990 | 0.990 | 0.000 | 9.500 | 0.95 |

_set/K = mean set size / number of classes; rows are demoted below the divider._
"""

_NEW_HEADER_BOARD = """\
| dataset | method | marginal | worst_class | frac_undercovered | frac_sig_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10c_contrast_s3 | oracle | 0.900 | 0.850 | 0.300 | 0.000 | 1.500 | 0.15 |
| cifar100c_contrast_s3 | split | 0.880 | 0.400 | 0.700 | 0.550 | 3.000 | 0.03 |
| ntu60_xsub | mondrian | 0.700 | 0.100 | 0.900 | ? | 2.000 | 0.03 |
| cifar10c_fog_s5 | weighted | 0.820 | 0.600 | 0.500 | 0.200 | 1.200 | 0.12 |
| *(vacuous below)* | | | | | | | | |
| cifar10c_contrast_s3 | acp | 0.990 | 0.990 | 0.000 | 0.000 | 9.500 | 0.95 |

_set/K = mean set size / number of classes; rows are demoted below the divider._
"""


def test_analyze_leaderboard_parses_old_and_new_header_to_the_same_rows():
    al = _load_analyze()
    old_rows = al.parse_leaderboard(_OLD_HEADER_BOARD)
    new_rows = al.parse_leaderboard(_NEW_HEADER_BOARD)
    assert len(old_rows) == 5
    assert new_rows == old_rows
    assert [r.mean_set_size for r in new_rows] == [1.5, 3.0, 2.0, 1.2, 9.5]
    assert [r.demoted for r in new_rows] == [False, False, False, False, True]
    assert al.exhibit_regime_labels(new_rows) == al.exhibit_regime_labels(old_rows)


def test_analyze_leaderboard_parses_a_board_rendered_by_to_markdown():
    al = _load_analyze()
    rows = [
        {**_row("cifar10c_contrast_s3", "split", 0.4, mean_set_size=3.0, n_classes=10,
                marginal=0.88), "frac_sig_undercovered": 0.25},
        _row("cifar10c_contrast_s3", "mondrian", 0.6, mean_set_size=2.0, n_classes=10),
    ]
    parsed = al.parse_leaderboard(to_markdown(aggregate(rows)))
    assert [(r.method, r.worst_class, r.mean_set_size, r.set_over_k) for r in parsed] == [
        ("mondrian", 0.6, 2.0, 0.2), ("split", 0.4, 3.0, 0.3)]
