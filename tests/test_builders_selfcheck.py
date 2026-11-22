"""Selfcheck tests for the vision cache builders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_cifar10c_clip_cache as cifar10c  # noqa: E402
import build_imagenetc_clip_cache as imagenetc  # noqa: E402

from pcpbench.datasets.convert import convert_legacy_npz  # noqa: E402
from pcpbench.datasets.shift_dataset import ShiftDataset  # noqa: E402
from pcpbench.runner import run  # noqa: E402


def test_no_heavy_deps_imported():
    """--selfcheck must never require torch/open_clip/datasets."""
    for mod in ("open_clip", "datasets"):
        assert mod not in sys.modules, f"{mod} was imported at module scope; selfcheck must stay lazy"


def test_cifar10c_selfcheck_main_returns_zero():
    rc = cifar10c.main(["--selfcheck"])
    assert rc == 0


def test_imagenetc_selfcheck_main_returns_zero():
    rc = imagenetc.main(["--selfcheck"])
    assert rc == 0


def test_imagenetc_selfcheck_logs_subset_line(capsys):
    rc = imagenetc.main(["--selfcheck"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUBSET:" in out
    assert "dropped the rest" in out


def test_cifar10c_cache_is_pcpbench_runnable(tmp_path):
    legacy_out = tmp_path / "cifar10c_legacy.npz"
    rc = cifar10c.main(["--selfcheck", "--selfcheck-out", str(legacy_out)])
    assert rc == 0

    z = np.load(legacy_out, allow_pickle=False)
    assert set(z.files) == {"source_logits", "source_labels", "target_logits", "target_labels"}
    assert z["source_logits"].shape[1] == 10

    pcp_out = tmp_path / "cifar10c_pcp.npz"
    convert_legacy_npz(legacy_out, pcp_out, name="cifar10c_selfcheck", shift_family="corruption")

    ds = ShiftDataset.from_npz(pcp_out)
    assert ds.n_classes == 10
    row = run(ds, "split", alpha=0.1, seeds=2)
    assert np.isfinite(row["worst_class"])


def test_imagenetc_cache_is_pcpbench_runnable(tmp_path):
    legacy_out = tmp_path / "imagenetc_legacy.npz"
    rc = imagenetc.main(["--selfcheck", "--selfcheck-out", str(legacy_out)])
    assert rc == 0

    z = np.load(legacy_out, allow_pickle=False)
    assert set(z.files) == {"source_logits", "source_labels", "target_logits", "target_labels"}
    assert z["source_logits"].shape[1] == 20

    pcp_out = tmp_path / "imagenetc_pcp.npz"
    convert_legacy_npz(legacy_out, pcp_out, name="imagenetc_selfcheck", shift_family="corruption")

    ds = ShiftDataset.from_npz(pcp_out)
    assert ds.n_classes == 20
    row = run(ds, "split", alpha=0.1, seeds=2)
    assert np.isfinite(row["worst_class"])
