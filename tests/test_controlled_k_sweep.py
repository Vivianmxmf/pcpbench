"""Tests for scripts/controlled_k_sweep.py (a standalone script, not part of the installed
``pcpbench`` package -- loaded here by file path, matching the convention in
test_analyze_leaderboard.py)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pcpbench.datasets.shift_dataset import SOURCE_CAL, TARGET_TEST
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.runner import METHODS
from pcpbench.scores.softmax_scores import lac_score_matrix, softmax

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "controlled_k_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("controlled_k_sweep", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cks = _load_module()


def test_restrict_to_subset_keeps_only_chosen_labels_and_relabels_contiguously():
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    subset = cks.draw_class_subset(k=6, n_total_classes=20, seed=1)
    restricted = cks.restrict_to_subset(ds, subset)

    orig_mask = np.isin(ds.labels, subset)
    assert restricted.logits.shape[0] == int(orig_mask.sum())

    assert set(np.unique(restricted.labels)) == set(range(6))
    assert restricted.n_classes == 6

    for j, orig_class in enumerate(subset):
        rows_of_j = restricted.labels == j
        orig_rows = ds.labels == orig_class
        assert rows_of_j.sum() == orig_rows.sum()
        np.testing.assert_array_equal(
            restricted.logits[rows_of_j], ds.logits[orig_rows][:, subset]
        )


def test_controlled_calibration_holds_per_class_size_constant_when_available():
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    subset = cks.draw_class_subset(k=8, n_total_classes=20, seed=2)
    restricted = cks.restrict_to_subset(ds, subset)

    idx, shortfalls = cks.controlled_calibration_indices(
        restricted.labels, restricted.split, k=8, n_per_class=50, seed=2
    )
    assert shortfalls == {}

    sub_split = restricted.split[idx]
    sub_labels = restricted.labels[idx]
    cal_labels = sub_labels[sub_split == SOURCE_CAL]
    for c in range(8):
        assert int((cal_labels == c).sum()) == 50

    assert int((sub_split == TARGET_TEST).sum()) == int((restricted.split == TARGET_TEST).sum())


def test_controlled_calibration_logs_shortfall_without_padding():
    ds = make_synthetic_shift(K=10, n_per_class=30, shift=0.6, seed=0)
    subset = cks.draw_class_subset(k=5, n_total_classes=10, seed=0)
    restricted = cks.restrict_to_subset(ds, subset)

    idx, shortfalls = cks.controlled_calibration_indices(
        restricted.labels, restricted.split, k=5, n_per_class=50, seed=0
    )
    assert shortfalls
    assert set(shortfalls) == set(range(5))
    assert all(deficit == 20 for deficit in shortfalls.values())

    sub_labels = restricted.labels[idx]
    sub_split = restricted.split[idx]
    cal_labels = sub_labels[sub_split == SOURCE_CAL]
    for c in range(5):
        assert int((cal_labels == c).sum()) == 30


def test_lac_scores_recomputed_over_k_columns_not_carried_from_wider_softmax():
    ds = make_synthetic_shift(K=20, n_per_class=50, shift=0.6, seed=0)
    subset = cks.draw_class_subset(k=6, n_total_classes=20, seed=3)
    restricted = cks.restrict_to_subset(ds, subset)

    assert restricted.logits.shape[1] == 6

    restricted_probs = softmax(restricted.logits)
    np.testing.assert_allclose(restricted_probs.sum(axis=1), 1.0, atol=1e-6)

    orig_mask = np.isin(ds.labels, subset)
    full_probs_sliced = softmax(ds.logits[orig_mask])[:, subset]
    assert not np.allclose(full_probs_sliced.sum(axis=1), 1.0, atol=1e-3)

    assert not np.allclose(restricted_probs, full_probs_sliced, atol=1e-6)

    scores = lac_score_matrix(restricted.logits)
    assert scores.shape == (restricted.logits.shape[0], 6)
    np.testing.assert_allclose(scores, 1.0 - restricted_probs, atol=1e-6)


def _capture_significance_inputs(monkeypatch, modules):
    from pcpbench.metrics import validity_floor as vf

    calls = []

    def spy(mask, labels, K, n_cal_per_class, alpha=0.1, q=0.05):
        n_eval = np.bincount(np.asarray(labels, dtype=np.int64), minlength=K)[:K]
        calls.append((n_eval, np.asarray(n_cal_per_class, dtype=np.int64).copy(), alpha, q))
        return vf.frac_significantly_undercovered(mask, labels, K, n_cal_per_class,
                                                  alpha=alpha, q=q)

    for module in modules:
        monkeypatch.setattr(module, "frac_significantly_undercovered", spy)
    return calls


def _check_implication(per_class_coverage, stored_sig, frac_under, n_eval, n_cal, alpha, q):
    from pcpbench.metrics.validity_floor import significance_class_status, significance_cutoff

    status = significance_class_status(n_eval, n_cal, alpha)
    flags = []
    for c in np.flatnonzero(status == 0):
        cov = per_class_coverage[c]
        flagged = int(round(cov * n_eval[c])) <= significance_cutoff(
            int(n_eval[c]), int(n_cal[c]), alpha, q)
        if flagged:
            assert cov < 1 - alpha
        flags.append(flagged)
    if not flags:
        assert np.isnan(stored_sig)
        return False
    assert stored_sig == pytest.approx(np.mean(flags))
    if np.all((status == 0) | (status == 1)):
        assert stored_sig <= frac_under
        return True
    return False


def test_sweep_runs_end_to_end_and_returns_one_worst_class_per_k_method():
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    k_values = [5, 10, 20]
    records = cks.run_sweep(
        ds, k_values=k_values, subset_seeds=2, n_cal_per_class=50, alpha=0.10,
        methods=sorted(METHODS),
    )
    summaries = cks.summarize(records)

    seen = {(s.k, s.method) for s in summaries}
    expected = {(k, m) for k in k_values for m in sorted(METHODS)}
    assert seen == expected

    for s in summaries:
        assert s.k in k_values
        assert np.isfinite(s.worst_class_mean)
        assert 0.0 <= s.worst_class_mean <= 1.0
        assert s.n_seeds == 2

    md = cks.render_markdown(summaries, k_values)
    assert "worst-class coverage vs K" in md
    for k in k_values:
        assert f"K={k}" in md


def test_draw_class_subset_is_deterministic_given_same_seed():
    a = cks.draw_class_subset(k=10, n_total_classes=100, seed=4)
    b = cks.draw_class_subset(k=10, n_total_classes=100, seed=4)
    np.testing.assert_array_equal(a, b)


def test_draw_class_subset_differs_across_seeds_generally():
    a = cks.draw_class_subset(k=10, n_total_classes=100, seed=0)
    b = cks.draw_class_subset(k=10, n_total_classes=100, seed=1)
    assert not np.array_equal(a, b)


def test_full_sweep_is_deterministic_given_same_seed():
    ds = make_synthetic_shift(K=12, n_per_class=60, shift=0.5, seed=0)
    r1 = cks.run_sweep(ds, k_values=[6], subset_seeds=2, n_cal_per_class=30,
                        alpha=0.10, methods=["split", "mondrian"])
    r2 = cks.run_sweep(ds, k_values=[6], subset_seeds=2, n_cal_per_class=30,
                        alpha=0.10, methods=["split", "mondrian"])
    assert r1 == r2


def test_draw_class_subset_rejects_k_larger_than_available():
    with pytest.raises(ValueError):
        cks.draw_class_subset(k=15, n_total_classes=10, seed=0)


def test_target_cal_pool_and_eval_disjoint_for_every_k_and_subset_seed():
    """The invariant that matters most: no method may be scored on any target row used to
    calibrate it."""
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    for k in (5, 10, 20):
        for subset_seed in range(3):
            sub, _ = cks.build_k_subproblem(ds, k, subset_seed, n_cal_per_class=30)
            tgt_logits, tgt_labels = sub.target_test()
            cal_idx, eval_idx = cks.split_target_labeled_and_eval(
                tgt_logits, tgt_labels, k, subset_seed, eval_frac=0.5
            )
            assert set(cal_idx.tolist()).isdisjoint(set(eval_idx.tolist()))
            assert len(cal_idx) > 0 and len(eval_idx) > 0
            assert len(cal_idx) + len(eval_idx) == len(tgt_labels)


def test_eval_only_dataset_excludes_cal_pool_rows():
    """``_build_eval_only_dataset`` must keep ALL source-cal rows but only the eval-index target
    rows -- no cal-pool row's logits may appear in the eval-only dataset's target view
    (content-level disjointness, not just a count check)."""
    ds = make_synthetic_shift(K=12, n_per_class=60, shift=0.5, seed=0)
    sub, _ = cks.build_k_subproblem(ds, k=6, subset_seed=1, n_cal_per_class=30)
    tgt_logits, tgt_labels = sub.target_test()
    cal_idx, eval_idx = cks.split_target_labeled_and_eval(
        tgt_logits, tgt_labels, k=6, seed=1, eval_frac=0.5
    )
    sub_eval = cks._build_eval_only_dataset(sub, eval_idx)

    assert int((sub_eval.split == SOURCE_CAL).sum()) == int((sub.split == SOURCE_CAL).sum())
    assert int((sub_eval.split == TARGET_TEST).sum()) == len(eval_idx)
    np.testing.assert_array_equal(sorted(sub_eval.target_test()[1]), sorted(tgt_labels[eval_idx]))

    cal_rows = {tuple(row) for row in tgt_logits[cal_idx]}
    eval_only_rows = {tuple(row) for row in sub_eval.target_test()[0]}
    assert cal_rows.isdisjoint(eval_only_rows)


def test_run_one_subproblem_acp_oracle_never_scored_on_calibration_rows():
    """End-to-end: run acp/oracle through the full per-(K,subset_seed) pipeline."""
    ds = make_synthetic_shift(K=16, n_per_class=100, shift=0.6, seed=0)
    records = cks.run_one_subproblem(
        ds, k=8, subset_seed=2, n_cal_per_class=40, alpha=0.10,
        methods=["acp", "oracle", "split"],
    )
    by_method = {r.method: r for r in records}
    n_restricted_target = by_method["split"].n_eval_total + by_method["acp"].n_target_cal_pool
    for method in ("acp", "oracle"):
        r = by_method[method]
        assert r.n_target_cal_pool > 0
        assert r.n_eval_total == by_method["split"].n_eval_total
        assert r.n_target_cal_pool + r.n_eval_total == n_restricted_target
    assert by_method["split"].n_target_cal_pool == 0


def test_low_count_eval_classes_detected_when_target_is_sparse():
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2])
    low = cks.low_count_eval_classes(labels, k=3, floor=5)
    assert low == (0, 2)


def test_low_count_eval_classes_empty_when_all_above_floor():
    labels = np.repeat(np.arange(4), 10)
    assert cks.low_count_eval_classes(labels, k=4, floor=5) == ()


def test_run_one_subproblem_reports_low_count_eval_classes_field():
    """Real CIFAR-100-C-scale synthetic data should NOT trigger the floor, but the field must
    exist and be zero, proving the guard runs."""
    ds = make_synthetic_shift(K=10, n_per_class=100, shift=0.5, seed=0)
    records = cks.run_one_subproblem(
        ds, k=10, subset_seed=0, n_cal_per_class=40, alpha=0.10, methods=["split"],
    )
    assert all(r.n_low_count_eval_classes == 0 for r in records)
_OLD_RECORD_FIELDS = (
    "k", "subset_seed", "method", "marginal", "worst_class", "frac_undercovered",
    "mean_set_size", "set_over_k", "n_cal_total", "n_shortfall_classes", "n_eval_total",
    "n_target_cal_pool", "n_low_count_eval_classes",
)
_OLD_SUMMARY_FIELDS = (
    "k", "method", "n_seeds", "worst_class_mean", "worst_class_std", "marginal_mean",
    "marginal_std", "set_over_k_mean", "set_over_k_std",
)


def test_sweep_records_new_fields_and_summaries(monkeypatch):
    import pcpbench.runner as runner_module

    calls = _capture_significance_inputs(monkeypatch, [cks, runner_module])
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    records = cks.run_sweep(ds, k_values=[5, 10], subset_seeds=2, n_cal_per_class=50,
                            alpha=0.10, methods=sorted(METHODS))
    assert len(calls) == len(records)
    sufficient = 0
    for r, (n_eval, n_cal, alpha, q) in zip(records, calls):
        cov = [np.nan if c is None else c for c in r.per_class_coverage]
        sufficient += _check_implication(cov, r.frac_sig_undercovered,
                                         r.frac_undercovered, n_eval, n_cal, alpha, q)
    assert sufficient == len(records)
    field_names = [f.name for f in dataclasses.fields(cks.SweepRecord)]
    assert field_names[:len(_OLD_RECORD_FIELDS)] == list(_OLD_RECORD_FIELDS)
    assert field_names[len(_OLD_RECORD_FIELDS):] == ["frac_sig_undercovered",
                                                     "per_class_coverage"]
    for r in records:
        assert isinstance(r.frac_sig_undercovered, float)
        assert 0.0 <= r.frac_sig_undercovered <= 1.0
        assert len(r.per_class_coverage) == r.k
        assert min(c for c in r.per_class_coverage if c is not None) == pytest.approx(
            r.worst_class)
    summaries = cks.summarize(records)
    assert list(cks.SweepSummary._fields[:len(_OLD_SUMMARY_FIELDS)]) == list(
        _OLD_SUMMARY_FIELDS)
    for s in summaries:
        group = [r.frac_sig_undercovered for r in records
                 if r.k == s.k and r.method == s.method]
        assert s.frac_sig_undercovered_mean == pytest.approx(np.mean(group))
        assert s.frac_sig_undercovered_std == pytest.approx(np.std(group))
    payload = cks.build_json_payload("synthetic", [5, 10], 2, 50, 0.1, records, summaries)
    assert "frac_sig_undercovered" in payload["records"][0]
    assert "per_class_coverage" in payload["records"][0]
    assert "frac_sig_undercovered_mean" in payload["summary"][0]
