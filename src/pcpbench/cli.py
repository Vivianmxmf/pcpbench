"""Command-line entry point: ``pcpbench run --dataset ... --method ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .datasets.shift_dataset import ShiftDataset
from .datasets.synthetic import make_synthetic_shift
from .runner import METHODS, run
from .scores.matrix import SCORES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcpbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a calibrator on a dataset.")
    run_parser.add_argument(
        "--dataset", required=True,
        help="Path to a ShiftDataset .npz cache, or the literal 'synthetic'.",
    )
    run_parser.add_argument(
        "--method", required=True, choices=sorted(METHODS),
        help="Calibrator to run.",
    )
    run_parser.add_argument("--alpha", type=float, default=0.1)
    run_parser.add_argument("--seeds", type=int, default=5)
    run_parser.add_argument("--m-labeled", type=int, default=0)
    run_parser.add_argument(
        "--score", default="lac", choices=SCORES,
        help="Nonconformity score of split, mondrian, weighted, ppi, oracle and "
             "clustered (acp keeps its own APS score). A row run under a score "
             "other than lac is named <method>_<score>.",
    )
    run_parser.add_argument(
        "--out", default=None,
        help="Optional output JSON path, written exactly as given. The grid "
             "script names it outputs/<dataset>_<method>_<score>.json when the "
             "score is not lac.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.dataset == "synthetic":
            ds = make_synthetic_shift()
        else:
            ds = ShiftDataset.from_npz(args.dataset)

        row = run(ds, args.method, alpha=args.alpha, seeds=args.seeds,
                  m_labeled=args.m_labeled, score=args.score)

        text = json.dumps(row, indent=2)
        print(text)
        if args.out:
            Path(args.out).write_text(text)
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
