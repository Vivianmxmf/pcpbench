import numpy as np
import pytest

from pcpbench.datasets.convert import convert_legacy_npz, main
from pcpbench.datasets.shift_dataset import ShiftDataset
from pcpbench.runner import run


def _write_legacy_npz(path, seed=0, n_source=120, n_target=80, K=5):
    rng = np.random.default_rng(seed)
    source_logits = rng.normal(size=(n_source, K)).astype(np.float32)
    target_logits = rng.normal(size=(n_target, K)).astype(np.float32)
    source_labels = rng.integers(0, K, size=n_source)
    source_labels[:K] = np.arange(K)
    target_labels = rng.integers(0, K, size=n_target)
    target_labels[:K] = np.arange(K)
    np.savez(
        path,
        source_logits=source_logits,
        source_labels=source_labels,
        target_logits=target_logits,
        target_labels=target_labels,
    )


def test_convert_roundtrip(tmp_path):
    path_in = tmp_path / "legacy.npz"
    path_out = tmp_path / "converted.npz"
    _write_legacy_npz(path_in, n_source=120, n_target=80, K=5)

    convert_legacy_npz(path_in, path_out, name="synth", shift_family="test")

    ds = ShiftDataset.from_npz(path_out)
    sc_x, sc_y = ds.source_cal()
    tt_x, tt_y = ds.target_test()
    assert len(sc_x) == len(sc_y) == 120
    assert len(tt_x) == len(tt_y) == 80
    assert ds.n_classes == 5
    assert ds.meta["name"] == "synth"
    assert ds.meta["shift_family"] == "test"
    assert ds.meta["n_classes"] == 5
    assert ds.meta["source"] == "converted-legacy-4key"


def test_convert_end_to_end_runnable(tmp_path):
    path_in = tmp_path / "legacy.npz"
    path_out = tmp_path / "converted.npz"
    _write_legacy_npz(path_in, n_source=150, n_target=100, K=6)

    convert_legacy_npz(path_in, path_out, name="synth2", shift_family="test2")

    ds = ShiftDataset.from_npz(path_out)
    row = run(ds, "split", alpha=0.1, seeds=2)
    assert np.isfinite(row["worst_class"])


def test_convert_missing_key_raises(tmp_path):
    path_in = tmp_path / "legacy_missing.npz"
    path_out = tmp_path / "converted.npz"
    rng = np.random.default_rng(1)
    np.savez(
        path_in,
        source_logits=rng.normal(size=(10, 3)).astype(np.float32),
        source_labels=rng.integers(0, 3, size=10),
        target_logits=rng.normal(size=(10, 3)).astype(np.float32),
    )

    with pytest.raises(KeyError):
        convert_legacy_npz(path_in, path_out, name="x", shift_family="y")


def test_cli_main(tmp_path):
    path_in = tmp_path / "legacy.npz"
    path_out = tmp_path / "converted.npz"
    _write_legacy_npz(path_in, n_source=50, n_target=30, K=4)

    rc = main([
        "--in", str(path_in),
        "--out", str(path_out),
        "--name", "cliname",
        "--shift-family", "clifam",
    ])
    assert rc == 0
    ds = ShiftDataset.from_npz(path_out)
    assert ds.meta["name"] == "cliname"
