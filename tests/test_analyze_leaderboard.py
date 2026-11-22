"""Tests for scripts/analyze_leaderboard.py (a standalone stdlib script, not part of the
installed ``pcpbench`` package -- loaded here by file path)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_leaderboard", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


al = _load_module()


def test_recover_k_cifar10c_row():
    assert al._recover_k(2.0, 0.20) == 10


def test_recover_k_cifar100c_row():
    assert al._recover_k(50.0, 0.50) == 100


def test_recover_k_none_when_set_over_k_zero():
    assert al._recover_k(0.0, 0.0) is None


_K_SCALING_TABLE = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_gaussian_noise_s3 | acp | 0.900 | 0.700 | 0.200 | 2.0 | 0.20 |
| cifar100c_gaussian_noise_s3 | acp | 0.900 | 0.100 | 0.800 | 50.0 | 0.50 |
"""


def test_k_scaling_matched_pair_delta_and_emission(tmp_path):
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(_K_SCALING_TABLE)
    rows = al.parse_leaderboard(leaderboard.read_text())

    cells, summaries = al.exhibit_k_scaling(rows)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.method == "acp"
    assert cell.corruption == "gaussian_noise"
    assert cell.severity == 3
    assert cell.low_k == 10
    assert cell.high_k == 100
    assert cell.worst_class_low_k == pytest.approx(0.70)
    assert cell.worst_class_high_k == pytest.approx(0.10)
    assert cell.delta_worst == pytest.approx(-0.60)

    assert cell.informative is True

    assert len(summaries) == 1
    assert summaries[0].method == "acp"
    assert summaries[0].n_cells_all == 1
    assert summaries[0].mean_delta_worst_all == pytest.approx(-0.60)
    assert summaries[0].n_cells_informative == 1
    assert summaries[0].mean_delta_worst_informative == pytest.approx(-0.60)


def test_k_scaling_requires_at_least_two_k_values(tmp_path):
    single_k_table = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_gaussian_noise_s3 | acp | 0.900 | 0.700 | 0.200 | 2.0 | 0.20 |
"""
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(single_k_table)
    rows = al.parse_leaderboard(leaderboard.read_text())

    cells, summaries = al.exhibit_k_scaling(rows)

    assert cells == []
    assert summaries == []


def test_is_trivially_vacuous_true_when_full_and_near_full_set():
    assert al.is_trivially_vacuous(1.00, 0.95) is True


def test_is_trivially_vacuous_false_when_near_full_set_but_missing_a_class():
    assert al.is_trivially_vacuous(0.00, 0.95) is False


def test_is_trivially_vacuous_false_when_set_small():
    assert al.is_trivially_vacuous(0.97, 0.30) is False


def test_is_trivially_vacuous_false_when_not_near_full():
    assert al.is_trivially_vacuous(1.00, 0.85) is False


_K_SCALING_SATURATION_TABLE = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_defocus_blur_s3 | acp | 0.900 | 0.002 | 0.998 | 2.0 | 0.20 |
| cifar100c_defocus_blur_s3 | acp | 1.000 | 1.000 | 0.000 | 100.0 | 1.00 |
| cifar10c_jpeg_s3 | acp | 0.900 | 0.400 | 0.400 | 2.0 | 0.20 |
| cifar100c_jpeg_s3 | acp | 0.980 | 0.100 | 0.300 | 95.0 | 0.95 |
"""


def test_k_scaling_saturated_endpoint_excluded_from_informative_summary(tmp_path):
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(_K_SCALING_SATURATION_TABLE)
    rows = al.parse_leaderboard(leaderboard.read_text())

    cells, summaries = al.exhibit_k_scaling(rows)

    by_corruption = {c.corruption: c for c in cells}
    assert by_corruption["defocus_blur"].informative is False
    assert by_corruption["jpeg"].informative is True

    assert len(summaries) == 1
    s = summaries[0]
    assert s.method == "acp"
    assert s.n_cells_all == 2
    assert s.mean_delta_worst_all == pytest.approx((0.998 - 0.300) / 2)
    assert s.n_cells_informative == 1
    assert s.mean_delta_worst_informative == pytest.approx(-0.300)


def test_label_regime_a_marginal_holds_perclass_collapses():
    assert al.label_regime(marginal=0.88, worst_class=0.56) == al.REGIME_A


def test_label_regime_b_marginal_breaks():
    assert al.label_regime(marginal=0.36, worst_class=0.00) == al.REGIME_B


def test_label_regime_recovered():
    assert al.label_regime(marginal=0.95, worst_class=0.90) == al.RECOVERED


def test_exhibit_regime_labels_aggregates_non_demoted_rows(tmp_path):
    table = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | mondrian | 0.880 | 0.560 | 0.500 | 1.732 | 0.03 |
| ntu60_xsub | split | 0.360 | 0.000 | 1.000 | 0.594 | 0.06 |
| ntu60_xsub | oracle | 0.950 | 0.900 | 0.000 | 2.245 | 0.04 |
"""
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(table)
    rows = al.parse_leaderboard(leaderboard.read_text())

    regime = al.exhibit_regime_labels(rows)

    assert regime.overall_counts[al.REGIME_A] == 1
    assert regime.overall_counts[al.REGIME_B] == 1
    assert regime.overall_counts[al.RECOVERED] == 1
    assert all(item.family == "ntu60_xsub" for item in regime.per_family_method)


def test_monotonicity_contrast_increase_is_flagged(tmp_path):
    table = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_contrast_s1 | split | 0.900 | 0.700 | 0.100 | 1.0 | 0.10 |
| cifar10c_contrast_s2 | split | 0.700 | 0.400 | 0.300 | 1.0 | 0.10 |
| cifar10c_contrast_s3 | split | 0.900 | 0.900 | 0.000 | 1.0 | 0.10 |
"""
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(table)
    rows = al.parse_leaderboard(leaderboard.read_text())

    violations, n_dropped = al.exhibit_monotonicity(rows)

    assert len(violations) == 1
    v = violations[0]
    assert v.corruption == "contrast"
    assert v.method == "split"
    assert v.severity_low == 2
    assert v.severity_high == 3
    assert v.worst_class_low == pytest.approx(0.400)
    assert v.worst_class_high == pytest.approx(0.900)
    assert n_dropped == 0


def test_monotonicity_fog_increase_is_exempt(tmp_path):
    table = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_fog_s1 | split | 0.900 | 0.700 | 0.100 | 1.0 | 0.10 |
| cifar10c_fog_s2 | split | 0.700 | 0.400 | 0.300 | 1.0 | 0.10 |
| cifar10c_fog_s3 | split | 0.900 | 0.900 | 0.000 | 1.0 | 0.10 |
"""
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(table)
    rows = al.parse_leaderboard(leaderboard.read_text())

    violations, n_dropped = al.exhibit_monotonicity(rows)

    assert violations == []
    assert n_dropped == 0


_MONOTONICITY_SATURATION_TABLE = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10c_defocus_blur_s1 | acp | 0.900 | 0.700 | 0.100 | 1.0 | 0.10 |
| cifar10c_defocus_blur_s2 | acp | 0.850 | 0.600 | 0.150 | 1.5 | 0.15 |
| cifar10c_defocus_blur_s3 | acp | 0.800 | 0.500 | 0.200 | 2.0 | 0.20 |
| cifar10c_defocus_blur_s4 | acp | 0.750 | 0.400 | 0.300 | 3.0 | 0.30 |
| cifar10c_defocus_blur_s5 | acp | 1.000 | 1.000 | 0.000 | 10.0 | 1.00 |
"""


def test_monotonicity_skips_trivially_vacuous_severity(tmp_path):
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(_MONOTONICITY_SATURATION_TABLE)
    rows = al.parse_leaderboard(leaderboard.read_text())

    violations, n_dropped = al.exhibit_monotonicity(rows)

    assert violations == []
    assert n_dropped == 1


def test_parse_leaderboard_marks_rows_after_divider_as_demoted(tmp_path):
    table = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | split | 0.869 | 0.563 | 0.567 | 1.475 | 0.02 |
| *(vacuous below)* | | | | | | |
| ntu60_xsub | acp | 0.963 | 0.815 | 0.150 | 37.276 | 0.62 |

_set/K = mean set size / number of classes; rows are demoted below the divider._
"""
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(table)
    rows = al.parse_leaderboard(leaderboard.read_text())

    assert len(rows) == 2
    assert rows[0].method == "split"
    assert rows[0].demoted is False
    assert rows[1].method == "acp"
    assert rows[1].demoted is True


def test_parse_leaderboard_all_above_when_no_divider(tmp_path):
    table = """\
| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |
| --- | --- | --- | --- | --- | --- | --- |
| ntu60_xsub | split | 0.869 | 0.563 | 0.567 | 1.475 | 0.02 |
| ntu60_xsub | acp | 0.963 | 0.815 | 0.150 | 37.276 | 0.62 |
"""
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(table)
    rows = al.parse_leaderboard(leaderboard.read_text())

    assert len(rows) == 2
    assert all(row.demoted is False for row in rows)


def test_main_runs_against_real_leaderboard(tmp_path, capsys):
    real_leaderboard = Path(__file__).resolve().parents[1] / "leaderboard.md"
    out_path = tmp_path / "leaderboard_analysis.json"

    rc = al.main(["--leaderboard", str(real_leaderboard), "--out", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    captured = capsys.readouterr()
    assert "pcpbench leaderboard analysis" in captured.out


def test_kscaling_keeps_pairs_when_set_over_k_displays_zero():
    """Regression: K must come from the dataset family, not from round(mean_set/set_over_k)."""
    from analyze_leaderboard import exhibit_k_scaling, parse_leaderboard

    board = (
        "| dataset | method | marginal | worst_class | frac_undercovered | mean_set_size | set/K |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| cifar10c_contrast_s3 | split | 0.680 | 0.250 | 0.500 | 0.040 | 0.00 |\n"
        "| cifar100c_contrast_s3 | split | 0.688 | 0.032 | 0.900 | 4.000 | 0.04 |\n"
    )
    rows = parse_leaderboard(board)
    ks = {r.dataset: r.k for r in rows}
    assert ks["cifar10c_contrast_s3"] == 10
    assert ks["cifar100c_contrast_s3"] == 100
    cells, summaries = exhibit_k_scaling(rows)
    assert len(cells) == 1
    assert abs(cells[0].delta_worst - (0.032 - 0.250)) < 1e-9


def test_header_missing_a_required_column_raises(tmp_path):
    """A header row that lacks a required name raises instead of falling back to positions, and
    the message names the missing column."""
    board = (
        "| dataset | method | marginal | worst | frac_undercovered | mean_set_size | set/K |\n"
        "|---|---|---|---|---|---|---|\n"
        "| cifar10c_fog_s3 | split | 0.9 | 0.8 | 0.1 | 1.2 | 0.12 |\n"
    )
    with pytest.raises(ValueError, match="worst_class"):
        al.parse_leaderboard(board)
