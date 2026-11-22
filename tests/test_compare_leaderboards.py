"""Tests for scripts/compare_leaderboards.py, loaded by file path like the other standalone
scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS = _ROOT / "outputs" / "analysis"
_SCRIPT_PATH = _ROOT / "scripts" / "compare_leaderboards.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_leaderboards", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cl = _load_module()

_OLD_BOARD = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar100c_fog_s3 | split | 0.860 | 0.700 | 0.300 | 1.500 | 0.02 |
| cifar100c_fog_s3 | mondrian | 0.900 | 0.850 | 0.100 | 2.000 | 0.02 |
| *(vacuous below)* | | | | | | |
| cifar100c_fog_s3 | acp | 0.990 | 0.950 | 0.000 | 95.000 | 0.95 |

_set/K = mean set size / number of classes; rows are demoted below the divider._
"""

_NEW_BOARD = """\
| dataset | method | marginal | worst_class | frac_undercovered | frac_sig_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cifar100c_fog_s3 | mondrian | 0.900 | 0.850 | 0.100 | 0.010 | 2.000 | 0.02 |
| cifar100c_fog_s3 | split | 0.840 | 0.700 | 0.300 | 0.200 | 1.500 | 0.02 |
| cifar100c_fog_s3 | clustered | 0.880 | 0.800 | 0.200 | 0.050 | 1.800 | 0.02 |
| *(vacuous below)* | | | | | | | |
| cifar100c_fog_s3 | acp | 0.990 | 0.950 | 0.000 | 0.000 | 95.000 | 0.95 |

_set/K = mean set size / number of classes; rows are demoted below the divider._
"""


def test_fixture_boards_one_changed_cell_one_new_row_and_the_header_difference():
    payload = cl.compare_boards(_OLD_BOARD, _NEW_BOARD)
    assert payload["n_rows_old"] == 3
    assert payload["n_rows_new"] == 4
    assert payload["n_shared_rows"] == 3
    assert payload["n_changed_rows"] == 1
    assert payload["n_changed_cells"] == 1
    assert payload["changed_cells_per_column"] == {"marginal": 1}
    assert payload["rows_only_in_old"] == []
    assert payload["rows_only_in_new"] == [{"dataset": "cifar100c_fog_s3", "method": "clustered"}]
    assert payload["columns_only_in_new"] == ["frac_sig_undercovered"]
    assert payload["columns_only_in_old"] == []
    assert "frac_sig_undercovered" not in payload["columns_compared"]
    (row,) = payload["changed_rows"]
    assert (row["dataset"], row["method"]) == ("cifar100c_fog_s3", "split")
    cell = row["cells"]["marginal"]
    assert cell["old"] == "0.860" and cell["new"] == "0.840"
    assert cell["abs_diff"] == pytest.approx(0.02)
    assert row["regime_old"] == cl.al.REGIME_A
    assert row["regime_new"] == cl.al.REGIME_OTHER
    assert row["regime_changed"] is True
    assert payload["n_regime_changes"] == 1
    assert row["demoted_old"] is False and row["demoted_new"] is False
    assert payload["demotion_changes"] == []


def test_a_row_only_in_the_old_board_and_a_non_numeric_cell():
    new = _OLD_BOARD.replace("| cifar100c_fog_s3 | mondrian | 0.900 | 0.850 | 0.100 | 2.000 | 0.02 |\n", "")
    payload = cl.compare_boards(_OLD_BOARD, new)
    assert payload["rows_only_in_old"] == [{"dataset": "cifar100c_fog_s3", "method": "mondrian"}]
    assert payload["n_changed_cells"] == 0
    assert cl._abs_diff("0.100", "?") is None
    assert cl._abs_diff("nan", "0.1") is None


def test_identical_boards_give_zero_changes():
    payload = cl.compare_boards(_OLD_BOARD, _OLD_BOARD)
    assert payload["n_changed_rows"] == 0
    assert payload["n_changed_cells"] == 0
    assert payload["changed_cells_per_column"] == {}
    assert payload["n_shared_rows"] == 3


def test_duplicate_keys_raise():
    dup = _OLD_BOARD.replace(
        "| cifar100c_fog_s3 | mondrian |", "| cifar100c_fog_s3 | split |"
    )
    with pytest.raises(ValueError, match="duplicate"):
        cl.compare_boards(dup, _OLD_BOARD)


def test_cli_writes_only_out_and_prints_a_summary(tmp_path, capsys):
    old = tmp_path / "old.md"
    new = tmp_path / "new.md"
    old.write_text(_OLD_BOARD)
    new.write_text(_NEW_BOARD)
    out = tmp_path / "sub" / "diff.json"
    assert cl.main(["--old", str(old), "--new", str(new), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["n_changed_cells"] == 1
    assert data["old"] == str(old) and data["new"] == str(new)
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["diff.json", "new.md", "old.md"]
    printed = capsys.readouterr().out
    assert "1 of the 3 shared rows" in printed
    assert "—" not in printed and "–" not in printed


def test_the_submitted_board_against_the_committed_board_reproduces_the_committed_diff():
    """The submitted board against the committed board reproduces the committed diff."""
    submitted = (_ANALYSIS / "leaderboard_submitted.md").read_text(encoding="utf-8")
    committed_diff = json.loads((_ANALYSIS / "leaderboard_rerun_diff.json").read_text())
    committed = (_ROOT / "leaderboard.md").read_text()
    payload = cl.compare_boards(submitted, committed)
    labels = {"old", "new"}
    assert {k: v for k, v in payload.items() if k not in labels} == {
        k: v for k, v in committed_diff.items() if k not in labels
    }
    assert payload["rows_only_in_old"] == []
    assert payload["n_shared_rows"] == payload["n_rows_old"] > 0
    relabelled = [row for row in payload["changed_rows"] if row["regime_changed"]]
    assert len(relabelled) == payload["n_regime_changes"]
    assert {row["method"] for row in relabelled} <= {"acp"}
    assert {row["method"] for row in payload["demotion_changes"]} <= {"acp"}
    assert payload["rows_only_in_new"]
    assert {row["method"] for row in payload["rows_only_in_new"]} == {"clustered"}
