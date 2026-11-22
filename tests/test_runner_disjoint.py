"""Tests for the disjoint pool/eval held-out split in ``pcpbench.runner.run``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.runner import METHODS, _split_pool_and_eval, run


def _ds(K=8, n_per_class=200, shift=0.6, noise=0.3, seed=0):
    return make_synthetic_shift(K=K, n_per_class=n_per_class, shift=shift,
                                 noise=noise, seed=seed)


def test_split_pool_and_eval_is_disjoint_exhaustive_and_roughly_balanced():
    ds = _ds(K=8, n_per_class=200)
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes

    pool_idx, eval_idx = _split_pool_and_eval(tgt_logits, tgt_labels, K, seed=0)

    assert set(pool_idx.tolist()).isdisjoint(set(eval_idx.tolist()))
    union = np.sort(np.concatenate([pool_idx, eval_idx]))
    np.testing.assert_array_equal(union, np.arange(len(tgt_labels)))

    pool_labels = tgt_labels[pool_idx]
    eval_labels = tgt_labels[eval_idx]
    for c in range(K):
        n_pool_c = int((pool_labels == c).sum())
        n_eval_c = int((eval_labels == c).sum())
        assert n_pool_c + n_eval_c == n_per_class_for(ds, c)
        assert abs(n_pool_c - n_eval_c) <= 1


def n_per_class_for(ds, c):
    _, tgt_labels = ds.target_test()
    return int((tgt_labels == c).sum())


def test_all_seven_methods_produce_finite_worst_class():
    ds = _ds(K=8, n_per_class=200, shift=0.5, noise=0.3)
    assert set(METHODS) == {"split", "mondrian", "weighted", "ppi", "acp", "oracle",
                            "clustered"}
    for method in METHODS:
        row = run(ds, method, alpha=0.1, seeds=5)
        assert np.isfinite(row["worst_class"]), f"{method}: worst_class not finite"
        assert 0.0 <= row["worst_class"] <= 1.0
        assert np.isfinite(row["marginal"])
        assert np.isfinite(row["mean_set_size"])
        assert row["n_eval"] > 0
        assert row["n_pool"] > 0


class _RecordingFitted:
    def __init__(self, inner, record):
        self._inner = inner
        self._record = record

    def predict_set(self, test_logits):
        self._record["eval_logits"] = np.asarray(test_logits)
        return self._inner.predict_set(test_logits)

    def thresholds(self):
        return self._inner.thresholds()


class _RecordingCalibrator:
    def __init__(self, inner, records: list[dict]):
        self._inner = inner
        self._records = records

    def fit(self, cal_logits, cal_labels, **kwargs):
        record: dict = {"target_labeled": kwargs.get("target_labeled")}
        self._records.append(record)
        fitted = self._inner.fit(cal_logits, cal_labels, **kwargs)
        return _RecordingFitted(fitted, record)


def _make_recording_factory(real_cls, records: list[dict]):
    def factory():
        return _RecordingCalibrator(real_cls(), records)
    return factory


@pytest.mark.parametrize("method", ["oracle", "acp", "ppi"])
def test_labeled_budget_never_overlaps_the_eval_set_it_is_scored_against(
    monkeypatch, method,
):
    ds = _ds(K=8, n_per_class=200, shift=0.5, noise=0.3)
    real_cls = METHODS[method]
    records: list[dict] = []
    monkeypatch.setitem(METHODS, method, _make_recording_factory(real_cls, records))

    row = run(ds, method, alpha=0.1, seeds=5)
    assert np.isfinite(row["worst_class"])
    assert len(records) == 5

    for rec in records:
        target_labeled = rec["target_labeled"]
        eval_logits = rec["eval_logits"]
        if target_labeled is None:
            continue
        labeled_logits, _labeled_labels = target_labeled
        if len(labeled_logits) == 0:
            continue
        labeled_rows = {tuple(row_vals) for row_vals in labeled_logits}
        eval_rows = {tuple(row_vals) for row_vals in eval_logits}
        assert labeled_rows.isdisjoint(eval_rows), (
            f"{method}: a labeled-pool row leaked into the eval set"
        )


def test_different_seeds_give_different_eval_splits_and_worst_class_values():
    ds = _ds(K=8, n_per_class=200, shift=0.6, noise=0.35)
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes

    _, eval_idx_0 = _split_pool_and_eval(tgt_logits, tgt_labels, K, seed=0)
    _, eval_idx_1 = _split_pool_and_eval(tgt_logits, tgt_labels, K, seed=1)
    assert not np.array_equal(np.sort(eval_idx_0), np.sort(eval_idx_1))

    row = run(ds, "split", alpha=0.1, seeds=10)
    from pcpbench.metrics import worst_class_coverage

    src_logits, src_labels = ds.source_cal()
    per_seed_worst = []
    for seed in range(10):
        pool_idx, eval_idx = _split_pool_and_eval(tgt_logits, tgt_labels, K, seed)
        eval_logits, eval_labels = tgt_logits[eval_idx], tgt_labels[eval_idx]
        fitted = METHODS["split"]().fit(src_logits, src_labels, alpha=0.1, seed=seed)
        mask = fitted.predict_set(eval_logits)
        per_seed_worst.append(worst_class_coverage(mask, eval_labels, K))

    assert len(set(np.round(per_seed_worst, 10))) > 1, (
        "worst_class was identical across all 10 seeds -- eval split is not resampling"
    )
    assert np.isfinite(row["worst_class"])


@pytest.mark.parametrize("method", ["split", "mondrian", "weighted", "ppi", "acp", "oracle"])
def test_run_is_deterministic_given_same_args(method):
    ds = _ds(K=8, n_per_class=200, shift=0.5, noise=0.3)
    row1 = run(ds, method, alpha=0.1, seeds=5)
    row2 = run(ds, method, alpha=0.1, seeds=5)
    assert row1 == row2 or _rows_equal_with_nan(row1, row2)


def _rows_equal_with_nan(row1: dict, row2: dict) -> bool:
    assert row1.keys() == row2.keys()
    for k in row1:
        v1, v2 = row1[k], row2[k]
        if isinstance(v1, float) and isinstance(v2, float) and np.isnan(v1) and np.isnan(v2):
            continue
        if v1 != v2:
            return False
    return True


def test_run_is_deterministic_with_m_labeled_budget():
    ds = _ds(K=8, n_per_class=200, shift=0.5, noise=0.3)
    for method in ("oracle", "acp", "ppi"):
        row1 = run(ds, method, alpha=0.1, seeds=5, m_labeled=10)
        row2 = run(ds, method, alpha=0.1, seeds=5, m_labeled=10)
        assert _rows_equal_with_nan(row1, row2)


def test_row_has_transparency_fields_and_existing_fields_preserved():
    ds = _ds(K=8, n_per_class=200, shift=0.5, noise=0.3)
    row = run(ds, "split", alpha=0.1, seeds=4)
    for key in (
        "dataset", "method", "alpha", "seeds", "m_labeled", "n_classes",
        "marginal", "marginal_std", "worst_class", "worst_class_std",
        "frac_undercovered", "frac_undercovered_std",
        "mean_set_size", "mean_set_size_std",
        "n_eval", "n_pool",
    ):
        assert key in row, f"missing key {key}"
    assert row["method"] == "split"
    assert row["seeds"] == 4
    assert row["m_labeled"] == 0
    assert row["n_classes"] == 8
    _, tgt_labels = ds.target_test()
    assert row["n_eval"] + row["n_pool"] == pytest.approx(len(tgt_labels), abs=1)


def test_oracle_with_empty_pool_records_nan_not_crash():
    """With exactly 1 target example per class, ``_split_pool_and_eval``'s eval_frac=0.5 rule
    takes that single example into ``eval`` and leaves ``pool`` empty for every class (n_eval =
    max(1, round(1*0.5))=1, capped at len==1)."""
    ds = _ds(K=4, n_per_class=1, shift=0.5, noise=0.3)
    tgt_logits, tgt_labels = ds.target_test()
    pool_idx, eval_idx = _split_pool_and_eval(tgt_logits, tgt_labels, ds.n_classes, seed=0)
    assert len(pool_idx) == 0

    for method in ("oracle", "acp"):
        row = run(ds, method, alpha=0.3, seeds=3)
        assert np.isnan(row["worst_class"])
        assert np.isnan(row["marginal"])
        assert row["n_pool"] == 0.0


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_sweep_split():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "label_complexity_sweep", _SCRIPTS / "label_complexity_sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.split_pool_and_eval


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
@pytest.mark.parametrize("eval_frac", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("n_per_class", [3, 4, 5, 7])
def test_runner_split_is_byte_identical_to_the_sweep_split(seed, eval_frac, n_per_class):
    sweep_split = _load_sweep_split()
    ds = _ds(K=6, n_per_class=n_per_class, seed=seed)
    tgt_logits, tgt_labels = ds.target_test()
    K = ds.n_classes
    p1, e1 = _split_pool_and_eval(tgt_logits, tgt_labels, K, seed, eval_frac)
    p2, e2 = sweep_split(tgt_logits, tgt_labels, K, seed, eval_frac)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(e1, e2)
    assert p1.dtype == p2.dtype and e1.dtype == e2.dtype
    assert p1.tobytes() == p2.tobytes() and e1.tobytes() == e2.tobytes()


def test_eval_frac_one_gives_empty_pool_and_full_eval():
    ds = _ds(K=8, n_per_class=50)
    N = int((ds.split == 2).sum())
    for method in ("split", "mondrian", "weighted", "ppi"):
        row = run(ds, method, alpha=0.1, seeds=3, eval_frac=1.0)
        assert row["n_pool"] == 0.0
        assert int(round(row["n_eval"])) == N
        assert row["n_failed_seeds"] == 0
        assert row["n_valid_seeds"] == 3


@pytest.mark.parametrize("kwargs", [
    {"seeds": 0},
    {"seeds": -1},
    {"seeds": 1.5},
    {"seeds": float("nan")},
    {"seeds": True},
    {"m_labeled": -1},
    {"m_labeled": 1.5},
    {"m_labeled": True},
    {"eval_frac": 0.0},
    {"eval_frac": 1.5},
    {"eval_frac": float("nan")},
    {"alpha": 0.0},
    {"alpha": 1.0},
    {"alpha": float("nan")},
])
def test_run_rejects_invalid_protocol_parameters(kwargs):
    ds = _ds(K=4, n_per_class=30)
    base = {"alpha": 0.1, "seeds": 2}
    base.update(kwargs)
    with pytest.raises(ValueError):
        run(ds, "split", **base)


def test_empty_pool_records_failed_seed_counts_not_a_silent_partial():
    """When oracle/acp have no labeled pool for any seed, every seed is a recorded failure (not a
    swallowed NaN), and metrics are NaN with an explicit n_valid_seeds==0 so a caller can reject
    the row."""
    ds = _ds(K=4, n_per_class=1, shift=0.5, noise=0.3)
    for method in ("oracle", "acp"):
        row = run(ds, method, alpha=0.3, seeds=3)
        assert row["n_valid_seeds"] == 0
        assert row["n_failed_seeds"] == 3
        assert np.isnan(row["worst_class"])


def test_unexpected_calibrator_error_is_not_swallowed(monkeypatch):
    """A genuine calibrator bug must propagate, never become a NaN cell."""
    ds = _ds(K=4, n_per_class=40)

    class _Boom:
        def fit(self, *a, **k):
            raise ValueError("genuine bug, must propagate")

    monkeypatch.setitem(METHODS, "split", _Boom)
    with pytest.raises(ValueError, match="genuine bug"):
        run(ds, "split", alpha=0.1, seeds=2)
