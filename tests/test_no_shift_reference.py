"""Tests for scripts/no_shift_reference.py (a standalone script, not part of the installed
``pcpbench`` package -- loaded here by file path, matching the convention in
test_controlled_k_sweep.py) and for the two every-cache job scripts, scripts/run_ksweep_all.sh
and its sbatch wrapper, plus scripts/run_noshift_grid.sbatch."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pcpbench.datasets.shift_dataset import SOURCE_CAL, ShiftDataset
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.validity_floor import expected_min_coverage
from pcpbench.runner import METHODS, run

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_SCRIPT_PATH = _SCRIPTS / "no_shift_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("no_shift_reference", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nsr = _load_module()


def test_every_method_runs_and_the_exact_calibrators_hit_the_floor():
    """The floor bound."""
    ds = make_synthetic_shift()
    alpha = 0.10
    seeds = 10
    payload = nsr.build_payload(ds, sorted(METHODS), alpha, seeds, eval_frac=0.5,
                                 cache_path="synthetic")

    assert set(METHODS) == {
        "split", "mondrian", "weighted", "ppi", "acp", "oracle", "clustered",
    }
    by_method = {s["method"]: s for s in payload["summary"]}
    assert set(by_method) == set(METHODS)

    for method in ("split", "mondrian", "clustered", "oracle"):
        s = by_method[method]
        assert s["floor_status"] == "ok", f"{method}: floor_status={s['floor_status']!r}"
        assert s["floor_sd"] > 0.0
        se = s["floor_sd"] / math.sqrt(s["floor_n_seeds"])
        assert s["worst_class_mean"] >= s["floor_mean"] - 3.0 * se, (
            f"{method}: worst_class_mean={s['worst_class_mean']!r} floor_mean="
            f"{s['floor_mean']!r} floor_sd={s['floor_sd']!r} floor_n_seeds="
            f"{s['floor_n_seeds']!r}"
        )
        if method in ("mondrian", "oracle"):
            assert abs(s["worst_class_mean"] - s["floor_mean"]) <= 3.0 * se, (
                f"{method}: worst_class_mean={s['worst_class_mean']!r} floor_mean="
                f"{s['floor_mean']!r} floor_sd={s['floor_sd']!r} floor_n_seeds="
                f"{s['floor_n_seeds']!r}"
            )
        assert 0.87 <= s["marginal_mean"] <= 0.93, (
            f"{method}: marginal_mean={s['marginal_mean']!r} not in [0.87, 0.93]"
        )

    per_seed_by_method: dict[str, list] = {}
    for rec in payload["per_seed"]:
        per_seed_by_method.setdefault(rec["method"], []).append(rec)
    for method in METHODS:
        assert len(per_seed_by_method[method]) == seeds, (
            f"{method}: expected {seeds} per_seed entries, got "
            f"{len(per_seed_by_method[method])}"
        )


def _degenerate_no_budget_dataset(K: int = 3) -> ShiftDataset:
    rng = np.random.default_rng(0)
    labels = np.arange(K)
    n = len(labels)
    return ShiftDataset(
        logits=rng.normal(size=(n, K)).astype(np.float32),
        labels=labels,
        split=np.ones(n, dtype=np.int8),
        domain=np.zeros(n, dtype=np.int32),
        meta={"name": "degenerate_no_budget"},
    )


def _degenerate_zero_sd_dataset(K: int = 3) -> ShiftDataset:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(K), 4)
    n = len(labels)
    return ShiftDataset(
        logits=rng.normal(size=(n, K)).astype(np.float32),
        labels=labels,
        split=np.ones(n, dtype=np.int8),
        domain=np.zeros(n, dtype=np.int32),
        meta={"name": "degenerate_zero_sd"},
    )


def test_floor_status_no_successful_seed_when_every_seed_fails():
    ds = _degenerate_no_budget_dataset()
    payload = nsr.build_payload(ds, ["oracle"], 0.10, 4, eval_frac=0.5, cache_path="deg")
    s = payload["summary"][0]
    assert s["floor_status"] == "no_successful_seed"
    assert math.isnan(s["floor_mean"])
    assert math.isnan(s["floor_sd"])
    assert math.isnan(s["z"])
    assert payload["floor"]["oracle"]["status"] == "no_successful_seed"
    assert payload["floor"]["oracle"]["n_successful_seeds"] == 0
    for rec in payload["per_seed"]:
        assert rec["row"]["n_cal_per_class_per_seed"] == [None]
        assert math.isnan(rec["floor_mean"])
        assert math.isnan(rec["floor_sd"])


def test_floor_status_zero_sd_when_every_calibration_count_is_degenerate():
    ds = _degenerate_zero_sd_dataset()
    payload = nsr.build_payload(ds, ["mondrian"], 0.10, 3, eval_frac=0.5, cache_path="deg")
    s = payload["summary"][0]
    assert s["floor_status"] == "zero_sd"
    assert s["floor_mean"] == 1.0
    assert s["floor_sd"] == 0.0
    assert math.isnan(s["z"])
    assert payload["floor"]["mondrian"]["status"] == "zero_sd"
    assert payload["floor"]["mondrian"]["n_successful_seeds"] == 3


def _zero_count_classes_dataset() -> ShiftDataset:
    rng = np.random.default_rng(0)
    labels = np.concatenate([[0], np.repeat([2, 3, 4], 40)])
    n = len(labels)
    return ShiftDataset(
        logits=rng.normal(size=(n, 5)).astype(np.float32),
        labels=labels,
        split=np.full(n, SOURCE_CAL, dtype=np.int8),
        domain=np.zeros(n, dtype=np.int32),
        meta={"name": "zero_count_classes"},
    )


@pytest.mark.parametrize("method", ["split", "mondrian"])
def test_floor_skips_classes_without_calibration_or_evaluation_points(method):
    ds = _zero_count_classes_dataset()
    seeds = 3
    payload = nsr.build_payload(ds, [method], 0.10, seeds, eval_frac=0.5, cache_path="zero")
    assert len(payload["per_seed"]) == seeds
    for rec in payload["per_seed"]:
        cal = np.asarray(rec["row"]["n_cal_per_class_per_seed"][0])
        ev = np.asarray(rec["row"]["n_eval_per_class_per_seed"][0])
        assert cal[0] == 0 and ev[0] == 1
        assert cal[1] == 0 and ev[1] == 0
        keep = (cal > 0) & (ev > 0)
        assert keep.tolist() == [False, False, True, True, True]
        expected = expected_min_coverage(
            3, cal[keep], ev[keep], 0.10, seed=rec["view_seed"]
        )
        assert rec["floor_mean"] == expected.mean
        assert rec["floor_sd"] == expected.sd
        assert rec["floor_n_classes"] == 3
        assert rec["floor_n_no_cal_classes"] == 1
        assert rec["floor_n_no_eval_classes"] == 1
    assert payload["floor"][method]["status"] == "ok"
    assert payload["floor"][method]["n_successful_seeds"] == seeds
    assert payload["floor"][method]["max_no_cal_classes"] == 1
    assert payload["floor"][method]["max_no_eval_classes"] == 1


def test_floor_is_one_with_zero_sd_when_no_class_has_both_counts():
    row = {
        "n_cal_per_class_per_seed": [[0, 0, 0]],
        "n_eval_per_class_per_seed": [[1, 1, 0]],
    }
    est = nsr._floor_estimate_for_row(row, 3, 0.10, 0)
    assert est.mean == 1.0
    assert est.sd == 0.0
    assert nsr._floor_class_counts(row) == (0, 1, 2)


def test_floor_is_unchanged_where_every_class_has_both_counts():
    ds = make_synthetic_shift()
    payload = nsr.build_payload(ds, ["mondrian"], 0.10, 2, eval_frac=0.5, cache_path="synthetic")
    K = ds.n_classes
    for rec in payload["per_seed"]:
        cal = np.asarray(rec["row"]["n_cal_per_class_per_seed"][0], dtype=np.int64)
        ev = np.asarray(rec["row"]["n_eval_per_class_per_seed"][0], dtype=np.int64)
        assert (cal > 0).all() and (ev > 0).all()
        expected = expected_min_coverage(K, cal, ev, 0.10, seed=rec["view_seed"])
        assert rec["floor_mean"] == expected.mean
        assert rec["floor_sd"] == expected.sd
        assert rec["floor_n_classes"] == K
        assert rec["floor_n_no_cal_classes"] == 0
        assert rec["floor_n_no_eval_classes"] == 0
    assert payload["floor"]["mondrian"]["max_no_cal_classes"] == 0
    assert payload["floor"]["mondrian"]["max_no_eval_classes"] == 0


def _unequal_zero_counts_dataset() -> ShiftDataset:
    rng = np.random.default_rng(1)
    labels = np.concatenate([[0, 1], np.repeat([3, 4, 5], 40)])
    n = len(labels)
    return ShiftDataset(
        logits=rng.normal(size=(n, 6)).astype(np.float32),
        labels=labels,
        split=np.full(n, SOURCE_CAL, dtype=np.int8),
        domain=np.zeros(n, dtype=np.int32),
        meta={"name": "unequal_zero_counts"},
    )


@pytest.mark.parametrize("method", ["split", "mondrian"])
def test_each_count_key_holds_its_own_kind(method):
    payload = nsr.build_payload(
        _unequal_zero_counts_dataset(), [method], 0.10, 2, eval_frac=0.5, cache_path="unequal"
    )
    for rec in payload["per_seed"]:
        assert rec["floor_n_classes"] == 3
        assert rec["floor_n_no_eval_classes"] == 1
        assert rec["floor_n_no_cal_classes"] == 2
    assert payload["floor"][method]["max_no_eval_classes"] == 1
    assert payload["floor"][method]["max_no_cal_classes"] == 2


def test_floor_skips_a_class_with_calibration_points_and_no_evaluation_point():
    """The runner never produces this pair; the filter is a guard, and this test exercises it
    directly."""
    row = {
        "n_cal_per_class_per_seed": [[5, 20, 20]],
        "n_eval_per_class_per_seed": [[0, 10, 10]],
    }
    est = nsr._floor_estimate_for_row(row, 3, 0.10, 0)
    expected = expected_min_coverage(2, np.array([20, 20]), np.array([10, 10]), 0.10, seed=0)
    assert est.mean == expected.mean
    assert est.sd == expected.sd
    assert nsr._floor_class_counts(row) == (2, 1, 0)


def test_clustered_on_a_k1000_view_reads_a_zero_sd_floor():
    """The ImageNet path: an ImageNet-shaped cache with K 1000 and 50 source calibration rows per
    class."""
    K, per = 1000, 50
    rng = np.random.default_rng(123)
    labels = np.repeat(np.arange(K), per)
    logits = rng.normal(size=(K * per, K)).astype(np.float32)
    logits[np.arange(K * per), labels] += 3.2
    ds = ShiftDataset(
        logits=logits,
        labels=labels,
        split=np.full(K * per, SOURCE_CAL, dtype=np.int8),
        domain=np.zeros(K * per, dtype=np.int32),
        meta={"name": "k1000_shape"},
    )
    payload = nsr.build_payload(ds, ["clustered"], 0.10, 1, eval_frac=0.5, cache_path="k1000")
    (rec,) = payload["per_seed"]
    cal = np.asarray(rec["row"]["n_cal_per_class_per_seed"][0])
    ev = np.asarray(rec["row"]["n_eval_per_class_per_seed"][0])
    assert ev.min() == 12 and ev.max() == 12
    assert (cal == 0).sum() > 100
    assert cal.max() <= 8
    assert rec["floor_n_no_cal_classes"] == int((cal == 0).sum())
    assert rec["floor_n_no_eval_classes"] == 0
    assert rec["floor_n_classes"] == K - int((cal == 0).sum())
    assert rec["floor_mean"] == 1.0 and rec["floor_sd"] == 0.0
    s = payload["summary"][0]
    assert s["floor_status"] == "zero_sd"
    assert math.isnan(s["z"])
    assert payload["floor"]["clustered"]["max_no_cal_classes"] == int((cal == 0).sum())
    assert payload["floor"]["clustered"]["max_no_eval_classes"] == 0


def test_output_json_has_no_nan_or_infinity_token_on_disk(tmp_path):
    """A strict JSON reader (one that rejects the NaN/Infinity/-Infinity constants Python's own
    ``json`` module accepts as a non-standard extension) must still be able to parse the whole
    file, on a payload that mixes an ``ok``, a ``no_successful_seed`` and a ``zero_sd`` method
    (each one's own NaN-bearing fields, including the embedded runner row's raw aggregate fields
    on the failed seeds, must all have been written as JSON null)."""
    out_path = tmp_path / "deg.json"
    ds_budget = _degenerate_no_budget_dataset()
    ds_zero_sd = _degenerate_zero_sd_dataset()

    payload_ok = nsr.build_payload(make_synthetic_shift(), ["split"], 0.10, 3,
                                    eval_frac=0.5, cache_path="synthetic")
    payload_no_seed = nsr.build_payload(ds_budget, ["oracle"], 0.10, 3,
                                         eval_frac=0.5, cache_path="deg_budget")
    payload_zero_sd = nsr.build_payload(ds_zero_sd, ["mondrian"], 0.10, 3,
                                         eval_frac=0.5, cache_path="deg_zero_sd")

    def _reject_constant(token):
        raise ValueError(f"non-finite JSON constant on disk: {token!r}")

    for payload in (payload_ok, payload_no_seed, payload_zero_sd):
        text = json.dumps(nsr._json_safe(payload), indent=2)
        assert "NaN" not in text, text
        assert "Infinity" not in text, text
        json.loads(text, parse_constant=_reject_constant)

    nsr.main([
        "--cache", "synthetic", "--methods", "oracle", "--alpha", "0.10",
        "--seeds", "3", "--out", str(out_path),
    ])
    on_disk = out_path.read_text()
    json.loads(on_disk, parse_constant=_reject_constant)


def test_calibration_halves_are_disjoint_stratified_and_cover_source_cal_exactly():
    ds = make_synthetic_shift(K=10, n_per_class=201, shift=0.6, seed=0)
    src_logits, src_labels = ds.source_cal()
    K = ds.n_classes

    a_idx, b_idx = nsr.split_calibration_halves(src_logits, src_labels, K, seed=3)

    assert set(a_idx.tolist()).isdisjoint(set(b_idx.tolist()))
    union = np.sort(np.concatenate([a_idx, b_idx]))
    np.testing.assert_array_equal(union, np.arange(len(src_labels)))

    a_labels = src_labels[a_idx]
    b_labels = src_labels[b_idx]
    for c in range(K):
        n_a = int((a_labels == c).sum())
        n_b = int((b_labels == c).sum())
        assert n_a + n_b == 201
        assert abs(n_a - n_b) <= 1


def test_build_no_shift_view_keeps_a_as_source_cal_and_b_as_target_test():
    ds = make_synthetic_shift(K=6, n_per_class=40, shift=0.6, seed=1)
    src_logits, src_labels = ds.source_cal()
    K = ds.n_classes
    a_idx, b_idx = nsr.split_calibration_halves(src_logits, src_labels, K, seed=0)

    view = nsr.build_no_shift_view(ds, a_idx, b_idx)

    view_cal_logits, view_cal_labels = view.source_cal()
    view_tgt_logits, view_tgt_labels = view.target_test()

    np.testing.assert_array_equal(view_cal_logits, src_logits[a_idx])
    np.testing.assert_array_equal(view_cal_labels, src_labels[a_idx])
    np.testing.assert_array_equal(view_tgt_logits, src_logits[b_idx])
    np.testing.assert_array_equal(view_tgt_labels, src_labels[b_idx])
    assert len(view.labels) == len(a_idx) + len(b_idx)


def test_per_seed_record_matches_a_direct_runner_call_on_the_same_view():
    ds = make_synthetic_shift(K=8, n_per_class=200, shift=0.6, seed=0)
    alpha = 0.10
    eval_frac = 0.5
    src_logits, src_labels = ds.source_cal()
    K = ds.n_classes

    a_idx, b_idx = nsr.split_calibration_halves(src_logits, src_labels, K, seed=0)
    view_0 = nsr.build_no_shift_view(ds, a_idx, b_idx)
    direct_row = run(view_0, "mondrian", alpha, seeds=1, m_labeled=0, eval_frac=eval_frac)

    payload = nsr.build_payload(ds, ["mondrian"], alpha, seeds=3, eval_frac=eval_frac,
                                 cache_path="synthetic")
    rec0 = next(r for r in payload["per_seed"] if r["method"] == "mondrian" and r["view_seed"] == 0)

    assert rec0["row"].keys() == direct_row.keys()
    for key in direct_row:
        v1, v2 = rec0["row"][key], direct_row[key]
        if isinstance(v1, float) and isinstance(v2, float) and np.isnan(v1) and np.isnan(v2):
            continue
        assert v1 == v2, f"key {key!r} differs: {v1!r} != {v2!r}"


@pytest.mark.parametrize("name", [
    "run_ksweep_all.sh", "run_ksweep_all.sbatch", "run_noshift_grid.sbatch",
])
def test_bash_n_passes(name):
    script = _SCRIPTS / name
    assert script.exists(), f"missing {script}"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n {name} failed: {result.stderr}"


_FAKE_PY = """#!/usr/bin/env bash
echo "FAKE_PY_CALLED $*"
prev=""
for arg in "$@"; do
    if [ "$prev" = "--out" ]; then
        mkdir -p "$(dirname "$arg")"
        echo '{}' > "$arg"
    fi
    prev="$arg"
done
"""


def test_run_ksweep_all_dry_run_one_invocation_per_cache_then_skip_on_rerun(tmp_path):
    fake_py = tmp_path / "fake_py.sh"
    fake_py.write_text(_FAKE_PY)
    fake_py.chmod(0o755)

    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "cifar100c_contrast_s3.npz").write_bytes(b"")
    (caches / "cifar100c_gaussian_noise_s5.npz").write_bytes(b"")

    outputs = tmp_path / "outputs"

    env = {
        "PY": str(fake_py),
        "CACHES": str(caches),
        "PATH": "/usr/bin:/bin",
    }

    first = subprocess.run(
        ["bash", str(_SCRIPTS / "run_ksweep_all.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    calls = [l for l in first.stdout.splitlines() if l.startswith("FAKE_PY_CALLED")]
    assert len(calls) == 2, first.stdout
    for line in calls:
        assert "--k-values 10,25,50,100" in line
        assert "--subset-seeds 5" in line
        assert "--n-cal-per-class 50" in line
        assert "--alpha 0.10" in line
        assert "--eval-frac 0.5" in line
    assert (outputs / "analysis" / "controlled_k_sweep_contrast_s3.json").exists()
    assert (outputs / "analysis" / "controlled_k_sweep_gaussian_noise_s5.json").exists()
    assert not (outputs / "analysis" / "controlled_k_sweep_cifar100c_contrast_s3.json").exists()
    assert not (outputs / "analysis" / "controlled_k_sweep_cifar100c_gaussian_noise_s5.json").exists()

    second = subprocess.run(
        ["bash", str(_SCRIPTS / "run_ksweep_all.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "SKIP:" in second.stdout
    calls_2 = [l for l in second.stdout.splitlines() if l.startswith("FAKE_PY_CALLED")]
    assert len(calls_2) == 0, second.stdout


def test_run_ksweep_all_force_reruns(tmp_path):
    fake_py = tmp_path / "fake_py.sh"
    fake_py.write_text(_FAKE_PY)
    fake_py.chmod(0o755)

    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "cifar100c_contrast_s3.npz").write_bytes(b"")

    env = {
        "PY": str(fake_py),
        "CACHES": str(caches),
        "FORCE": "1",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(["bash", str(_SCRIPTS / "run_ksweep_all.sh")], cwd=tmp_path, env=env,
                    capture_output=True, text=True)
    second = subprocess.run(["bash", str(_SCRIPTS / "run_ksweep_all.sh")], cwd=tmp_path,
                             env=env, capture_output=True, text=True)
    calls_2 = [l for l in second.stdout.splitlines() if l.startswith("FAKE_PY_CALLED")]
    assert len(calls_2) == 1, second.stdout


def test_run_ksweep_all_skips_a_committed_style_prefix_free_output(tmp_path):
    """A cache named ``caches/cifar100c_contrast_s3.npz`` must be recognised as already done by a
    pre-existing ``controlled_k_sweep_contrast_s3.json`` (the committed, prefix-free name), not
    re-run into a second file under the full-stem name."""
    fake_py = tmp_path / "fake_py.sh"
    fake_py.write_text(_FAKE_PY)
    fake_py.chmod(0o755)

    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "cifar100c_contrast_s3.npz").write_bytes(b"")

    analysis = tmp_path / "outputs" / "analysis"
    analysis.mkdir(parents=True)
    committed = analysis / "controlled_k_sweep_contrast_s3.json"
    committed.write_text('{"committed": true}')

    env = {"PY": str(fake_py), "CACHES": str(caches), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(_SCRIPTS / "run_ksweep_all.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP: contrast_s3" in result.stdout, result.stdout
    calls = [l for l in result.stdout.splitlines() if l.startswith("FAKE_PY_CALLED")]
    assert len(calls) == 0, result.stdout
    assert committed.read_text() == '{"committed": true}'
    assert not (analysis / "controlled_k_sweep_cifar100c_contrast_s3.json").exists()


_FAKE_PY_FAIL_CONTRAST = """#!/usr/bin/env bash
echo "FAKE_PY_CALLED $*"
case "$*" in
    *contrast_s3*) exit 3 ;;
esac
prev=""
for arg in "$@"; do
    if [ "$prev" = "--out" ]; then
        mkdir -p "$(dirname "$arg")"
        echo '{}' > "$arg"
    fi
    prev="$arg"
done
"""


def test_run_ksweep_all_a_failed_cache_is_logged_and_the_batch_continues(tmp_path):
    fake_py = tmp_path / "fake_py_fail.sh"
    fake_py.write_text(_FAKE_PY_FAIL_CONTRAST)
    fake_py.chmod(0o755)

    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "cifar100c_contrast_s3.npz").write_bytes(b"")
    (caches / "cifar100c_gaussian_noise_s5.npz").write_bytes(b"")

    env = {"PY": str(fake_py), "CACHES": str(caches), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(_SCRIPTS / "run_ksweep_all.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL: contrast_s3" in result.stdout, result.stdout
    calls = [l for l in result.stdout.splitlines() if l.startswith("FAKE_PY_CALLED")]
    assert len(calls) == 2, result.stdout
    assert (tmp_path / "outputs" / "analysis" / "controlled_k_sweep_gaussian_noise_s5.json").exists()
    assert not (tmp_path / "outputs" / "analysis" / "controlled_k_sweep_contrast_s3.json").exists()


def test_a_write_interrupted_midway_leaves_no_output_file(tmp_path, monkeypatch):
    """run_noshift_grid.sbatch skips every cache whose output exists, so a truncated file would
    be accepted by every later pass."""
    out_path = tmp_path / "noshift" / "synthetic.json"
    real_write_text = Path.write_text

    def _write_half_then_die(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise KeyboardInterrupt("time limit reached during the write")

    monkeypatch.setattr(Path, "write_text", _write_half_then_die)
    with pytest.raises(KeyboardInterrupt):
        nsr.main(["--cache", "synthetic", "--methods", "split", "--seeds", "1",
                  "--out", str(out_path)])
    monkeypatch.undo()
    assert not out_path.exists()
    assert sorted(p.name for p in out_path.parent.glob("*.json")) == []

    nsr.main(["--cache", "synthetic", "--methods", "split", "--seeds", "1",
              "--out", str(out_path)])
    assert json.loads(out_path.read_text())["summary"]
    assert sorted(p.name for p in out_path.parent.iterdir()) == ["synthetic.json"]
