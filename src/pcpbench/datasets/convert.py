"""Legacy 4-key cache converter: source/target logits+labels -> pcpbench ShiftDataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .shift_dataset import SOURCE_CAL, TARGET_TEST, ShiftDataset

_LEGACY_KEYS = ("source_logits", "source_labels", "target_logits", "target_labels")


def convert_legacy_npz(path_in, path_out, *, name: str, shift_family: str) -> None:
    """Convert a legacy 4-key npz cache into a pcpbench ``ShiftDataset`` npz."""
    z = np.load(Path(path_in), allow_pickle=False)
    for key in _LEGACY_KEYS:
        if key not in z.files:
            raise KeyError(f"{path_in} lacks '{key}'; expected a legacy 4-key cache "
                            f"with keys {_LEGACY_KEYS}.")

    source_logits = z["source_logits"]
    source_labels = z["source_labels"]
    target_logits = z["target_logits"]
    target_labels = z["target_labels"]

    n_source = len(source_labels)
    n_target = len(target_labels)

    logits = np.concatenate([source_logits, target_logits]).astype(np.float32)
    labels = np.concatenate([source_labels, target_labels]).astype(np.int64)
    split = np.concatenate([
        np.full(n_source, SOURCE_CAL, dtype=np.int8),
        np.full(n_target, TARGET_TEST, dtype=np.int8),
    ])
    domain = np.concatenate([
        np.zeros(n_source, dtype=np.int32),
        np.ones(n_target, dtype=np.int32),
    ])
    K = int(max(source_labels.max(), target_labels.max())) + 1
    meta = {
        "name": name,
        "shift_family": shift_family,
        "n_classes": K,
        "source": "converted-legacy-4key",
    }

    ds = ShiftDataset(logits=logits, labels=labels, split=split, domain=domain, meta=meta)
    ds.to_npz(path_out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcpbench.datasets.convert",
        description="Convert a legacy 4-key npz cache into a pcpbench ShiftDataset npz.",
    )
    parser.add_argument("--in", dest="path_in", required=True, help="Legacy 4-key npz input path.")
    parser.add_argument("--out", dest="path_out", required=True, help="pcpbench ShiftDataset npz output path.")
    parser.add_argument("--name", required=True, help="Dataset name to store in meta.")
    parser.add_argument("--shift-family", required=True, help="Shift family label to store in meta.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    convert_legacy_npz(args.path_in, args.path_out, name=args.name, shift_family=args.shift_family)

    ds = ShiftDataset.from_npz(args.path_out)
    n_source = len(ds.source_cal()[1])
    n_target = len(ds.target_test()[1])
    print(f"converted {args.path_in} -> {args.path_out}: "
          f"source={n_source} target={n_target} K={ds.n_classes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
