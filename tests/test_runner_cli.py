import json
import math

import numpy as np
import pytest

from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.validity_floor import significance_class_status, significance_cutoff
from pcpbench.runner import run, METHODS, SIG_Q, calibration_counts_per_class
from pcpbench.cli import main


def test_run_produces_metric_row():
    ds = make_synthetic_shift(K=6, n_per_class=300, shift=0.6, seed=0)
    row = run(ds, "mondrian", alpha=0.1, seeds=3)
    for k in ("marginal", "worst_class", "frac_undercovered", "mean_set_size"):
        assert k in row and np.isfinite(row[k])
    assert set(METHODS) >= {"split","mondrian","weighted","ppi","oracle"}


def test_cli_synthetic_writes_leaderboard(tmp_path):
    out = tmp_path / "lb.json"
    rc = main(["run","--dataset","synthetic","--method","split",
               "--alpha","0.1","--seeds","2","--out",str(out)])
    assert rc == 0
    row = json.loads(out.read_text())
    assert row["method"] == "split" and "worst_class" in row


class _FittedTopClass:
    def predict_set(self, test_logits):
        logits = np.asarray(test_logits)
        mask = np.zeros(logits.shape, dtype=bool)
        mask[np.arange(len(logits)), logits.argmax(axis=1)] = True
        return mask

    def thresholds(self):
        return np.zeros(0)


class _TopClass:
    def fit(self, cal_logits, cal_labels, *, alpha=0.1, seed=0):
        return _FittedTopClass()


def test_a_method_registered_in_methods_yields_a_row(monkeypatch, tmp_path):
    """A new calibrator with fit and predict_set, registered in METHODS, yields a row through run
    and through the command line, as the paper states."""
    monkeypatch.setitem(METHODS, "topclass", _TopClass)
    ds = make_synthetic_shift(K=4, n_per_class=100, shift=0.6, seed=0)
    row = run(ds, "topclass", alpha=0.1, seeds=2)
    reference = run(ds, "split", alpha=0.1, seeds=2)
    assert row["method"] == "topclass" and row["n_valid_seeds"] == 2
    assert row["mean_set_size"] == 1.0
    assert set(row) == set(reference)
    assert row["n_cal_per_class_per_seed"] == reference["n_cal_per_class_per_seed"]
    assert row["score"] is None and row["set_rule"] is None
    out = tmp_path / "row.json"
    assert main(["run", "--dataset", "synthetic", "--method", "topclass", "--seeds", "1",
                 "--out", str(out)]) == 0
    assert json.loads(out.read_text())["method"] == "topclass"
    with pytest.raises(ValueError, match=r"Unknown method 'nothing'; choose from \['acp'"):
        calibration_counts_per_class("nothing", None, np.array([0]), None, 1)


_ANCHOR_COMMON = {
    "dataset": "synthetic_shift", "alpha": 0.1, "seeds": 5, "n_valid_seeds": 5,
    "n_failed_seeds": 0, "m_labeled": 0, "n_classes": 12, "n_eval": 3000.0,
    "n_pool": 3000.0,
}
_ANCHOR_BASE = {
    "split": {
        "marginal": 0.5083333333333333, "marginal_std": 0.00388158043413591,
        "worst_class": 0.0024000000000000002, "worst_class_std": 0.0019595917942265423,
        "frac_undercovered": 1.0, "frac_undercovered_std": 0.0,
        "mean_set_size": 0.5083333333333333, "mean_set_size_std": 0.00388158043413591,
    },
    "mondrian": {
        "marginal": 0.5072666666666666, "marginal_std": 0.0035364765892999697,
        "worst_class": 0.0024000000000000002, "worst_class_std": 0.0019595917942265423,
        "frac_undercovered": 1.0, "frac_undercovered_std": 0.0,
        "mean_set_size": 0.5072666666666666, "mean_set_size_std": 0.0035364765892999697,
    },
    "weighted": {
        "marginal": 0.5079999999999999, "marginal_std": 0.0036086316273931727,
        "worst_class": 0.0024000000000000002, "worst_class_std": 0.0019595917942265423,
        "frac_undercovered": 1.0, "frac_undercovered_std": 0.0,
        "mean_set_size": 0.5079999999999999, "mean_set_size_std": 0.0036086316273931727,
    },
    "ppi": {
        "marginal": 0.8978666666666666, "marginal_std": 0.003215241480476673,
        "worst_class": 0.8712, "worst_class_std": 0.012237646832622694,
        "frac_undercovered": 0.5333333333333334, "frac_undercovered_std": 0.04082482904638632,
        "mean_set_size": 1.208, "mean_set_size_std": 0.009512564790259746,
    },
    "acp": {
        "marginal": 0.9853999999999999, "marginal_std": 0.0023981474331473294,
        "worst_class": 0.9232000000000001, "worst_class_std": 0.01371714255958577,
        "frac_undercovered": 0.0, "frac_undercovered_std": 0.0,
        "mean_set_size": 6.998600000000001, "mean_set_size_std": 0.058075582935803116,
    },
    "oracle": {
        "marginal": 0.8972, "marginal_std": 0.003283629428273268,
        "worst_class": 0.8592000000000001, "worst_class_std": 0.012749901960407393,
        "frac_undercovered": 0.55, "frac_undercovered_std": 0.04082482904638632,
        "mean_set_size": 1.2102, "mean_set_size_std": 0.009499941520287852,
    },
    "clustered": {
        "marginal": 0.5090666666666667, "marginal_std": 0.0042760313895532165,
        "worst_class": 0.0024000000000000002, "worst_class_std": 0.0019595917942265423,
        "frac_undercovered": 1.0, "frac_undercovered_std": 0.0,
        "mean_set_size": 0.5090666666666667, "mean_set_size_std": 0.0042760313895532165,
    },
}
_PER_SEED_KEYS = ("marginal", "worst_class", "frac_undercovered",
                  "frac_sig_undercovered", "mean_set_size",
                  "n_sig_untestable_classes", "n_sig_degenerate_classes")
_COUNT_KEYS = ("n_sig_untestable_classes", "n_sig_degenerate_classes")


def _flags_from_stored_vectors(cov, n_eval, n_cal, alpha=0.1, q=SIG_Q):
    n_eval = np.asarray(n_eval, dtype=np.int64)
    status = significance_class_status(n_eval, np.asarray(n_cal, dtype=np.int64), alpha)
    flags, covs = [], []
    for c in np.flatnonzero(status == 0):
        hits = int(round(cov[c] * n_eval[c]))
        flags.append(hits <= significance_cutoff(int(n_eval[c]), int(n_cal[c]), alpha, q))
        covs.append(cov[c])
    return status, flags, covs


def _assert_theorem_on_row(row, alpha=0.1):
    n_sufficient = 0
    for s_idx, cov in enumerate(row["per_class_coverage_per_seed"]):
        if cov is None:
            continue
        status, flags, covs = _flags_from_stored_vectors(
            cov, row["n_eval_per_class_per_seed"][s_idx],
            row["n_cal_per_class_per_seed"][s_idx], alpha)
        for flagged, c in zip(flags, covs):
            if flagged:
                assert c < 1 - alpha
        sig = row["per_seed"]["frac_sig_undercovered"][s_idx]
        if flags:
            assert sig == pytest.approx(np.mean(flags))
        else:
            assert sig is None
        assert row["per_seed"]["n_sig_untestable_classes"][s_idx] == int(
            np.count_nonzero(status == 2))
        assert row["per_seed"]["n_sig_degenerate_classes"][s_idx] == int(
            np.count_nonzero(status == 3))
        if flags and np.all((status == 0) | (status == 1)):
            n_sufficient += 1
            assert sig <= row["per_seed"]["frac_undercovered"][s_idx]
    return n_sufficient


@pytest.mark.parametrize("method", sorted(_ANCHOR_BASE))
def test_synthetic_row_keeps_anchor_values_and_adds_significance_column(method):
    seeds = 5
    ds = make_synthetic_shift()
    K = ds.n_classes
    row = run(ds, method, alpha=0.1, seeds=seeds)

    expected = {**_ANCHOR_COMMON, "method": method, **_ANCHOR_BASE[method]}
    for key, value in expected.items():
        assert row[key] == value, f"{method}: existing key {key} moved"

    assert "frac_sig_undercovered" in row and "frac_sig_undercovered_std" in row
    assert set(row["per_seed"]) == set(_PER_SEED_KEYS)
    for key in _PER_SEED_KEYS:
        assert len(row["per_seed"][key]) == seeds

    pcc = row["per_class_coverage_per_seed"]
    assert len(pcc) == seeds and all(len(v) == K for v in pcc)
    n_eval_pc = row["n_eval_per_class_per_seed"]
    assert len(n_eval_pc) == seeds
    assert all(sum(v) == row["n_eval"] for v in n_eval_pc)
    n_cal_pc = row["n_cal_per_class_per_seed"]
    assert len(n_cal_pc) == seeds and all(len(v) == K for v in n_cal_pc)

    assert _assert_theorem_on_row(row) == seeds
    for key in _COUNT_KEYS:
        assert row["per_seed"][key] == [0] * seeds
        assert row[f"{key}_per_seed"] == [0] * seeds
    assert row["frac_sig_undercovered"] == pytest.approx(
        np.mean(row["per_seed"]["frac_sig_undercovered"]))
    assert row["frac_undercovered"] == pytest.approx(
        np.mean(row["per_seed"]["frac_undercovered"]))
    assert row["worst_class"] == pytest.approx(np.mean([min(v) for v in pcc]))
    json.dumps({k: row[k] for k in ("per_seed", "per_class_coverage_per_seed",
                                     "n_eval_per_class_per_seed",
                                     "n_cal_per_class_per_seed",
                                     "n_sig_untestable_classes_per_seed",
                                     "n_sig_degenerate_classes_per_seed")},
               allow_nan=False)


def test_calibration_counts_follow_the_set_each_threshold_is_computed_from():
    ds = make_synthetic_shift()
    K = ds.n_classes
    row_split = run(ds, "split", alpha=0.1, seeds=2)
    row_oracle = run(ds, "oracle", alpha=0.1, seeds=2)
    row_ppi = run(ds, "ppi", alpha=0.1, seeds=2)
    row_acp = run(ds, "acp", alpha=0.1, seeds=2)
    row_clustered = run(ds, "clustered", alpha=0.1, seeds=2)
    for row in (row_split, row_acp):
        assert row["n_cal_per_class_per_seed"] == [[500] * K] * 2
    for row in (row_oracle, row_ppi):
        assert row["n_cal_per_class_per_seed"] == [[250] * K] * 2
    for counts in row_clustered["n_cal_per_class_per_seed"]:
        assert all(0 < c < 500 for c in counts)


def test_calibration_counts_helper_ppi_without_budget_uses_source_counts():
    src_labels = np.array([0, 0, 1, 2, 2, 2])
    counts = calibration_counts_per_class("ppi", None, src_labels, None, 3)
    assert counts.tolist() == [2, 1, 3]
    counts = calibration_counts_per_class("ppi", None, src_labels, np.array([1, 1]), 3)
    assert counts.tolist() == [0, 2, 0]
    counts = calibration_counts_per_class("oracle", None, src_labels, np.array([2]), 3)
    assert counts.tolist() == [0, 0, 1]
    for method in ("split", "mondrian", "weighted", "acp"):
        counts = calibration_counts_per_class(method, None, src_labels, np.array([2]), 3)
        assert counts.tolist() == [2, 1, 3]


def test_failed_seeds_store_null_entries_in_seed_order():
    ds = make_synthetic_shift(K=3, n_per_class=1, seed=0)
    row = run(ds, "oracle", alpha=0.1, seeds=2)
    assert row["n_failed_seeds"] == 2
    assert math.isnan(row["frac_sig_undercovered"])
    for key in _PER_SEED_KEYS:
        assert row["per_seed"][key] == [None, None]
    assert row["per_class_coverage_per_seed"] == [None, None]
    assert row["n_cal_per_class_per_seed"] == [None, None]
    for key in _COUNT_KEYS:
        assert row[f"{key}_per_seed"] == [None, None]
    assert row["n_eval_per_class_per_seed"] == [[1, 1, 1], [1, 1, 1]]
    row_ppi = run(ds, "ppi", alpha=0.1, seeds=1)
    assert row_ppi["n_cal_per_class_per_seed"] == [[1, 1, 1]]


def test_ppi_small_budget_reads_nan_with_every_class_degenerate():
    """At m 5 every class is calibrated on 5 labeled points, below the smallest non-degenerate
    count, so no class is tested and every seed reads NaN with the degenerate count equal to K."""
    ds = make_synthetic_shift()
    K = ds.n_classes
    row = run(ds, "ppi", alpha=0.1, seeds=2, m_labeled=5)
    assert math.isnan(row["frac_sig_undercovered"])
    assert math.isnan(row["frac_sig_undercovered_std"])
    assert row["per_seed"]["frac_sig_undercovered"] == [None, None]
    assert row["per_seed"]["n_sig_degenerate_classes"] == [K, K]
    assert row["n_sig_degenerate_classes_per_seed"] == [K, K]
    assert row["per_seed"]["n_sig_untestable_classes"] == [0, 0]
    assert row["n_sig_untestable_classes_per_seed"] == [0, 0]
    assert row["n_cal_per_class_per_seed"] == [[5] * K] * 2
    assert all(v is not None for v in row["per_seed"]["frac_undercovered"])
    _assert_theorem_on_row(row)


def test_ppi_budget_at_the_smallest_nondegenerate_count_is_tested():
    """At a budget equal to the smallest non-degenerate count every class is tested and the
    per-class implication holds on the stored vectors."""
    from pcpbench.metrics.validity_floor import coverage_law

    m = next(n for n in range(1, 1000) if coverage_law(n, 0.1)[1] >= 1)
    row = run(make_synthetic_shift(), "ppi", alpha=0.1, seeds=2, m_labeled=m)
    assert row["n_sig_degenerate_classes_per_seed"] == [0, 0]
    assert all(v is not None for v in row["per_seed"]["frac_sig_undercovered"])
    assert _assert_theorem_on_row(row) == 2
