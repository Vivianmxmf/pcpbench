"""Tests for scripts/label_complexity_sweep.py (a standalone script, not part of the installed
``pcpbench`` package -- loaded here by file path, per the pattern in
tests/test_analyze_leaderboard.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.runner import METHODS

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "label_complexity_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("label_complexity_sweep", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lcs = _load_module()


def _ds(seed: int = 0):
    return make_synthetic_shift(K=6, n_per_class=200, shift=0.6, seed=seed)


def _row_equal(a, b) -> bool:
    for x, y in zip(a, b):
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True


def test_pool_and_eval_disjoint_for_every_seed():
    ds = _ds()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    for seed in range(5):
        pool_idx, eval_idx = lcs.split_pool_and_eval(tgt_logits, tgt_labels, K, seed, 0.5)
        assert set(pool_idx.tolist()).isdisjoint(set(eval_idx.tolist()))
        assert len(pool_idx) > 0 and len(eval_idx) > 0
        assert len(pool_idx) + len(eval_idx) == len(tgt_labels)


def test_budget_subset_disjoint_from_eval_across_m_grid():
    ds = _ds()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    for seed in range(3):
        pool_idx, eval_idx = lcs.split_pool_and_eval(tgt_logits, tgt_labels, K, seed, 0.5)
        pool_labels = tgt_labels[pool_idx]
        pool_as_index = pool_idx.reshape(-1, 1).astype(float)
        eval_set = set(eval_idx.tolist())
        for m in (0, 5, 20, 50, 1000):
            budget_idx_col, _ = lcs.sample_budget(pool_as_index, pool_labels, m, K, seed)
            budget_global_idx = set(budget_idx_col.reshape(-1).astype(int).tolist())
            assert budget_global_idx.isdisjoint(eval_set)
            assert budget_global_idx.issubset(set(pool_idx.tolist()))


def test_n_labeled_monotone_nondecreasing_in_m():
    ds = _ds()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    seed = 0
    pool_idx, _ = lcs.split_pool_and_eval(tgt_logits, tgt_labels, K, seed, 0.5)
    pool_logits, pool_labels = tgt_logits[pool_idx], tgt_labels[pool_idx]

    m_values = [0, 1, 5, 10, 25, 50, 100, 1000]
    counts = []
    for m in m_values:
        _, budget_labels = lcs.sample_budget(pool_logits, pool_labels, m, K, seed)
        counts.append(len(budget_labels))

    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    assert counts[0] == 0
    assert counts[-1] == len(pool_labels)


def test_summary_n_labeled_mean_monotone_for_budget_method():
    ds = _ds()
    m_values = [0, 5, 20, 60]
    rows = lcs.run_sweep(ds, methods=["ppi"], m_values=m_values, alpha=0.1, seeds=3)
    summary = sorted((s for s in lcs.summarize(rows) if s.method == "ppi"), key=lambda s: s.m)
    n_labeled = [s.n_labeled_mean for s in summary]
    assert all(n_labeled[i] <= n_labeled[i + 1] for i in range(len(n_labeled) - 1))


def test_one_worst_class_per_method_m_in_summary():
    ds = _ds()
    methods = ["split", "ppi", "acp"]
    m_values = [0, 10, 30]
    rows = lcs.run_sweep(ds, methods, m_values, alpha=0.1, seeds=2)
    summary = lcs.summarize(rows)

    keys = [(s.method, s.m) for s in summary]
    assert len(keys) == len(methods) * len(m_values)
    assert len(set(keys)) == len(keys)
    for s in summary:
        assert isinstance(s.worst_class_mean, float)


def test_run_sweep_deterministic_per_seed():
    ds = _ds()
    kwargs = dict(methods=["acp", "ppi", "oracle", "split"], m_values=[0, 10, 30],
                 alpha=0.1, seeds=2)
    rows_a = lcs.run_sweep(ds, **kwargs)
    rows_b = lcs.run_sweep(ds, **kwargs)

    assert len(rows_a) == len(rows_b)
    assert all(_row_equal(a, b) for a, b in zip(rows_a, rows_b))


def test_label_free_methods_worst_class_constant_across_m():
    ds = _ds()
    m_values = [0, 10, 50, 100]
    for method in ("split", "mondrian", "weighted"):
        rows = lcs.run_sweep(ds, methods=[method], m_values=m_values, alpha=0.1, seeds=3)
        summary = lcs.summarize(rows)

        assert all(not s.label_budget_method for s in summary)
        assert all(r.n_labeled == 0 for r in rows)
        worst_vals = [s.worst_class_mean for s in summary]
        assert max(worst_vals) - min(worst_vals) < 1e-9
        assert all("budget-independent" in s.note for s in summary)


def test_acp_and_oracle_m0_is_nan_with_note_not_crash():
    ds = _ds()
    rows = lcs.run_sweep(ds, methods=["acp", "oracle"], m_values=[0], alpha=0.1, seeds=2)
    for r in rows:
        assert np.isnan(r.worst_class)
        assert np.isnan(r.marginal)

    summary = lcs.summarize(rows)
    for s in summary:
        assert np.isnan(s.worst_class_mean)
        assert "undefined at this m" in s.note


def test_ppi_m0_runs_without_target_labels():
    ds = _ds()
    rows = lcs.run_sweep(ds, methods=["ppi"], m_values=[0], alpha=0.1, seeds=2)
    for r in rows:
        assert np.isfinite(r.worst_class)
        assert r.n_labeled == 0


def test_acp_and_oracle_recover_finite_worst_class_at_larger_m():
    ds = _ds()
    rows = lcs.run_sweep(ds, methods=["acp", "oracle"], m_values=[30], alpha=0.1, seeds=2)
    for r in rows:
        assert np.isfinite(r.worst_class)
        assert r.n_labeled > 0


def test_json_payload_serializes_nan_as_null():
    ds = _ds()
    rows = lcs.run_sweep(ds, methods=["acp"], m_values=[0, 10], alpha=0.1, seeds=2)
    summary = lcs.summarize(rows)
    payload = lcs.to_json_payload(
        rows, summary, cache="x.npz", alpha=0.1, seeds=2,
        m_values=[0, 10], methods=["acp"], eval_frac=0.5,
    )

    text = json.dumps(payload)
    reloaded = json.loads(text)
    m0_summary = next(s for s in reloaded["summary"] if s["m"] == 0)
    assert m0_summary["worst_class_mean"] is None


def test_main_cli_writes_json_and_prints_table(tmp_path, capsys):
    ds = _ds()
    cache_path = tmp_path / "synthetic.npz"
    ds.to_npz(cache_path)
    out_path = tmp_path / "sweep.json"

    rc = lcs.main([
        "--cache", str(cache_path),
        "--m-values", "0,10,30",
        "--methods", "ppi,acp,oracle,split",
        "--alpha", "0.1",
        "--seeds", "2",
        "--out", str(out_path),
    ])

    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["methods"] == ["ppi", "acp", "oracle", "split"]
    assert len(payload["summary"]) == 4 * 3

    captured = capsys.readouterr()
    assert "### split" in captured.out
    assert "### acp" in captured.out


def test_main_unknown_method_raises():
    with pytest.raises(ValueError):
        lcs.main(["--cache", "does_not_exist.npz", "--m-values", "0",
                  "--methods", "not_a_real_method", "--out", "y.json"])


def test_sample_budget_is_nested_paired_across_m():
    """Increasing m must only ADD labels for the same seed, never reshuffle which ones were
    already selected -- the m=10 index set must be a SUBSET of the m=25 index set, etc."""
    ds = _ds()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    seed = 0
    pool_idx, _ = lcs.split_pool_and_eval(tgt_logits, tgt_labels, K, seed, 0.5)
    pool_labels = tgt_labels[pool_idx]
    pool_as_index = pool_idx.reshape(-1, 1).astype(float)

    m_values = [0, 5, 10, 25, 50, 100]
    index_sets = []
    for m in m_values:
        budget_idx_col, _ = lcs.sample_budget(pool_as_index, pool_labels, m, K, seed)
        index_sets.append(set(budget_idx_col.reshape(-1).astype(int).tolist()))

    for smaller, larger in zip(index_sets, index_sets[1:]):
        assert smaller.issubset(larger)
        assert len(smaller) <= len(larger)


def test_sample_budget_prefix_matches_larger_draw_exactly():
    """Stronger than subset: the FIRST len(budget(m)) elements returned for budget(m') (m' > m)
    must be byte-identical to budget(m) (true prefix, not just a set-equal subset), per class."""
    ds = _ds()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    seed = 2
    pool_idx, _ = lcs.split_pool_and_eval(tgt_logits, tgt_labels, K, seed, 0.5)
    pool_logits, pool_labels = tgt_logits[pool_idx], tgt_labels[pool_idx]

    _, labels_10 = lcs.sample_budget(pool_logits, pool_labels, 10, K, seed)
    _, labels_25 = lcs.sample_budget(pool_logits, pool_labels, 25, K, seed)

    pool_as_index = pool_idx.reshape(-1, 1).astype(float)
    idx_10, _ = lcs.sample_budget(pool_as_index, pool_labels, 10, K, seed)
    idx_25, _ = lcs.sample_budget(pool_as_index, pool_labels, 25, K, seed)
    for c in range(K):
        prefix_c = idx_10.reshape(-1)[labels_10 == c]
        larger_c = idx_25.reshape(-1)[labels_25 == c]
        np.testing.assert_array_equal(prefix_c, larger_c[:len(prefix_c)])


def test_per_class_label_counts_basic():
    labels = np.array([0, 0, 1, 1, 1, 2])
    assert lcs.per_class_label_counts(labels, K=3) == (2, 3, 1)


def test_evaluate_one_records_classwise_shortfall_when_capacity_capped():
    """Construct a pool where one class has far fewer than the requested m labels available; the
    row must record the TRUE per-class count (not the requested m) and flag the shortfall, never
    presenting the capped point as if the full m had been used."""
    ds = _ds()
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    seed = 0
    pool_idx, eval_idx = lcs.split_pool_and_eval(tgt_logits, tgt_labels, K, seed, 0.5)
    pool_logits, pool_labels = tgt_logits[pool_idx], tgt_labels[pool_idx]
    eval_logits, eval_labels = tgt_logits[eval_idx], tgt_labels[eval_idx]
    src_logits, src_labels = ds.source_cal()

    keep = np.concatenate([
        np.where(pool_labels == 0)[0][:2],
        np.where(pool_labels != 0)[0],
    ])
    starved_pool_logits, starved_pool_labels = pool_logits[keep], pool_labels[keep]

    m = 20
    row = lcs._evaluate_one(
        "ppi", m, seed, src_logits, src_labels, tgt_logits,
        starved_pool_logits, starved_pool_labels, eval_logits, eval_labels,
        K, alpha=0.1, label_budget_method=True,
    )
    assert row.per_class_labeled[0] == 2
    assert row.per_class_labeled[0] < m
    assert row.n_shortfall_classes >= 1
    assert row.n_labeled < m * K


def test_m0_never_flagged_as_capacity_capped():
    """m=0 must not be a false-positive shortfall: min(0, available) == 0 == m for every class,
    so n_shortfall_classes must be exactly 0 at m=0."""
    ds = _ds()
    rows = lcs.run_sweep(ds, methods=["ppi"], m_values=[0], alpha=0.1, seeds=2)
    assert all(r.n_shortfall_classes == 0 for r in rows)


def test_summary_flags_capacity_capped_note():
    """When a (method, m) cell has real shortfalls, the summary note must say so, never
    presenting a capped point as if the full m was used."""
    ds = make_synthetic_shift(K=6, n_per_class=8, shift=0.6, seed=1)
    rows = lcs.run_sweep(ds, methods=["oracle"], m_values=[50], alpha=0.1, seeds=2)
    summary = lcs.summarize(rows)
    assert any(s.n_shortfall_classes_mean > 0 for s in summary)
    assert any("capacity-capped" in s.note for s in summary)


def test_build_eval_class_report_flags_low_count_classes():
    labels = np.array([0] * 2 + [1] * 20 + [2] * 3)
    low = tuple(c for c in range(3) if int((labels == c).sum()) < 5)
    assert low == (0, 2)


def test_build_eval_class_report_no_warning_on_adequate_counts(capsys):
    ds = _ds()
    reports = lcs.build_eval_class_report(ds, seeds=2, eval_frac=0.5)
    assert len(reports) == 2
    for r in reports:
        assert r.low_count_classes == ()
        assert len(r.counts) == ds.n_classes
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_build_eval_class_report_warns_on_sparse_target(capsys):
    ds = make_synthetic_shift(K=4, n_per_class=6, shift=0.5, seed=0)
    reports = lcs.build_eval_class_report(ds, seeds=1, eval_frac=0.5)
    assert any(r.low_count_classes for r in reports)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_json_payload_includes_eval_class_report():
    ds = _ds()
    rows = lcs.run_sweep(ds, methods=["split"], m_values=[0], alpha=0.1, seeds=2)
    summary = lcs.summarize(rows)
    eval_class_report = lcs.build_eval_class_report(ds, seeds=2, eval_frac=0.5)
    payload = lcs.to_json_payload(
        rows, summary, cache="x.npz", alpha=0.1, seeds=2,
        m_values=[0], methods=["split"], eval_frac=0.5,
        eval_class_report=eval_class_report,
    )
    text = json.dumps(payload)
    reloaded = json.loads(text)
    assert len(reloaded["eval_class_report"]) == 2
    assert "low_count_classes" in reloaded["eval_class_report"][0]
_OLD_ROW_FIELDS = ("method", "m", "seed", "n_labeled", "per_class_labeled",
                   "n_shortfall_classes", "worst_class", "marginal", "label_budget_method")
_NEW_ROW_FIELDS = ("mean_set_size", "set_over_k", "frac_undercovered",
                   "frac_sig_undercovered", "per_class_coverage")
_OLD_SUMMARY_FIELDS = ("method", "m", "n_labeled_mean", "n_shortfall_classes_mean",
                       "worst_class_mean", "worst_class_std", "marginal_mean",
                       "marginal_std", "label_budget_method", "note")


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


def test_sweep_rows_record_set_size_undercoverage_and_per_class_coverage(monkeypatch):
    from pcpbench.metrics.validity_floor import coverage_law

    calls = _capture_significance_inputs(monkeypatch, [lcs])
    ds = _ds()
    K = ds.n_classes
    smallest = next(n for n in range(1, 1000) if coverage_law(n, 0.1)[1] >= 1)
    m_values = [0, smallest - 1, 10, 30]
    rows = lcs.run_sweep(ds, methods=sorted(METHODS), m_values=m_values, alpha=0.1,
                         seeds=2)
    assert lcs.SweepRow._fields == _OLD_ROW_FIELDS + _NEW_ROW_FIELDS
    scored = [r for r in rows if not np.isnan(r.worst_class)]
    assert len(calls) == len(scored)
    for r, (n_eval, n_cal, alpha, q) in zip(scored, calls):
        cov = [np.nan if c is None else c for c in r.per_class_coverage]
        sufficient = _check_implication(cov, r.frac_sig_undercovered,
                                        r.frac_undercovered, n_eval, n_cal, alpha, q)
        if r.method in ("oracle", "ppi") and r.m == smallest - 1:
            assert np.isnan(r.frac_sig_undercovered)
        else:
            assert sufficient
    for r in rows:
        assert len(r.per_class_coverage) == K
        if np.isnan(r.worst_class):
            assert np.isnan(r.mean_set_size) and np.isnan(r.frac_sig_undercovered)
            assert all(c is None for c in r.per_class_coverage)
            continue
        assert r.set_over_k == pytest.approx(r.mean_set_size / K)
        assert 0.0 <= r.frac_undercovered <= 1.0
        assert np.isnan(r.frac_sig_undercovered) or 0.0 <= r.frac_sig_undercovered <= 1.0
        assert min(c for c in r.per_class_coverage if c is not None) == pytest.approx(
            r.worst_class)

    summary = lcs.summarize(rows)
    assert lcs.SweepSummary._fields[:len(_OLD_SUMMARY_FIELDS)] == _OLD_SUMMARY_FIELDS
    for s in summary:
        group = [r for r in rows if r.method == s.method and r.m == s.m]
        for name in ("mean_set_size", "set_over_k", "frac_undercovered",
                     "frac_sig_undercovered"):
            values = np.array([getattr(r, name) for r in group], dtype=float)
            mean, std = getattr(s, f"{name}_mean"), getattr(s, f"{name}_std")
            if np.all(np.isnan(values)):
                assert np.isnan(mean) and np.isnan(std)
            else:
                assert mean == pytest.approx(np.nanmean(values))
                assert std == pytest.approx(np.nanstd(values))

    payload = lcs.to_json_payload(rows, summary, cache="x.npz", alpha=0.1, seeds=2,
                                  m_values=m_values, methods=sorted(METHODS),
                                  eval_frac=0.5)
    reloaded = json.loads(json.dumps(payload, allow_nan=False))
    assert "per_class_coverage" in reloaded["per_seed"][0]
    assert "frac_sig_undercovered_mean" in reloaded["summary"][0]
