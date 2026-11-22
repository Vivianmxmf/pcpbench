"""Rebuild the derived artifacts from the row files, the sweeps and the submitted board."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SUBMITTED_BOARD = "outputs/analysis/leaderboard_submitted.md"
RELEASED_BOARD = "leaderboard_released.md"
BOARD = "leaderboard.md"
BOARD_REFERENCE = "board_as_committed.md"
SWEEP_GLOB = "outputs/analysis/controlled_k_sweep_*.json"
REL_TOL = 1e-9
ABS_TOL = 1e-12
DIFFERENT, INCOMPLETE = 1, 2


class Step(NamedTuple):
    """One command and the files it writes: name in the work directory -> path in the repository."""

    title: str
    command: tuple[str, ...]
    writes: dict[str, str]
    needs_submitted_board: bool = False


class Rebuild(NamedTuple):
    """What the steps produced."""

    built: dict[str, Path]
    missing: dict[str, str]


def steps() -> tuple[Step, ...]:
    scripts = ROOT / "scripts"
    return (
        Step("the board", ("-m", "pcpbench.leaderboard", "--in", "outputs", "--out", BOARD,
                           "--datasets-from", BOARD_REFERENCE),
             {BOARD: BOARD}),
        Step("the cell-level difference from the submitted board",
             (str(scripts / "compare_leaderboards.py"), "--old", RELEASED_BOARD, "--new", BOARD,
              "--out", "leaderboard_rerun_diff.json"),
             {"leaderboard_rerun_diff.json": "outputs/analysis/leaderboard_rerun_diff.json"},
             needs_submitted_board=True),
        Step("the regime and K-scaling analysis of the headline",
             (str(scripts / "analyze_leaderboard.py"), "--leaderboard", BOARD, "--subset", "headline",
              "--out", "leaderboard_analysis.json"),
             {"leaderboard_analysis.json": "outputs/analysis/leaderboard_analysis.json"}),
        Step("the threshold sensitivity",
             (str(scripts / "threshold_sensitivity.py"), "--leaderboard", BOARD,
              "--json-out", "threshold_sensitivity.json", "--md-out", "THRESHOLD_SENSITIVITY.md",
              "--allow-count-change"),
             {"threshold_sensitivity.json": "outputs/analysis/threshold_sensitivity.json",
              "THRESHOLD_SENSITIVITY.md": "paper/camera_ready/THRESHOLD_SENSITIVITY.md"}),
        Step("the uncertainty of the controlled sweep and the label sweep",
             (str(scripts / "sweep_uncertainty.py"), "--analysis-dir", "outputs/analysis",
              "--out-json", "sweep_uncertainty.json", "--out-md", "SWEEP_UNCERTAINTY.md"),
             {"sweep_uncertainty.json": "outputs/analysis/sweep_uncertainty.json",
              "SWEEP_UNCERTAINTY.md": "paper/camera_ready/SWEEP_UNCERTAINTY.md"}),
        Step("the every-cache comparison with the matched reference",
             (str(scripts / "class_count_trend.py"), "--design", "independent",
              "--inputs-glob", SWEEP_GLOB,
              "--out-json", "all_cache_trend.json", "--out-md", "ALL_CACHE_TREND.md"),
             {"all_cache_trend.json": "outputs/analysis/all_cache_trend.json",
              "ALL_CACHE_TREND.md": "paper/camera_ready/ALL_CACHE_TREND.md"}),
    )


def submitted_board(root: Path) -> str | None:
    """The board of the submitted artifact, or None where its file is missing."""
    path = root / SUBMITTED_BOARD
    return path.read_text(encoding="utf-8") if path.is_file() else None


def prepare(work: Path, root: Path) -> bool:
    """Lay out the work directory."""
    (work / "outputs").symlink_to(root / "outputs", target_is_directory=True)
    shutil.copyfile(root / BOARD, work / BOARD_REFERENCE)
    released = submitted_board(root)
    if released is not None:
        (work / RELEASED_BOARD).write_text(released, encoding="utf-8")
    return released is not None


def run_steps(work: Path, root: Path, log: Callable[[str], None] = print) -> Rebuild:
    """Run every step in ``work``."""
    has_submitted_board = prepare(work, root)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(root / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    built: dict[str, Path] = {}
    missing: dict[str, str] = {}
    for step in steps():
        if step.needs_submitted_board and not has_submitted_board:
            reason = f"{SUBMITTED_BOARD} is missing"
            missing.update(dict.fromkeys(step.writes.values(), reason))
            log(f"SKIPPED {step.title}: {reason}")
            continue
        done = subprocess.run([sys.executable, *step.command], cwd=work, env=env,
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"{step.title} failed:\n{done.stdout}{done.stderr}")
        for name, destination in step.writes.items():
            if not (work / name).is_file():
                raise RuntimeError(f"{step.title} did not write {name}")
            built[destination] = work / name
        log(f"built {step.title}")
    return Rebuild(built, missing)


def same_json(old: object, new: object) -> bool:
    """Whether two parsed JSON values are equal apart from the last digits of a float."""
    if isinstance(old, dict) and isinstance(new, dict):
        return old.keys() == new.keys() and all(same_json(old[key], new[key]) for key in old)
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(same_json(a, b) for a, b in zip(old, new))
    if isinstance(old, float) and isinstance(new, float):
        return (math.isnan(old) and math.isnan(new)) or math.isclose(
            old, new, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return type(old) is type(new) and old == new


def same_file(destination: str, committed: Path, fresh: Path) -> bool:
    """Whether the committed file equals its rebuild: byte for byte, or, for a JSON file, apart
    from the last digits of a float."""
    if not committed.is_file():
        return False
    old, new = committed.read_bytes(), fresh.read_bytes()
    if old == new:
        return True
    if not destination.endswith(".json"):
        return False
    try:
        return same_json(json.loads(old), json.loads(new))
    except ValueError:
        return False


def differing(built: dict[str, Path], root: Path) -> list[str]:
    """The repository paths whose file is missing or differs from its rebuild."""
    return [destination for destination, fresh in sorted(built.items())
            if not same_file(destination, root / destination, fresh)]


def install(built: dict[str, Path], root: Path) -> list[str]:
    """Replace the repository files that differ from their rebuild."""
    changed = differing(built, root)
    staged: dict[str, Path] = {}
    try:
        for destination in changed:
            target = root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            staged[destination] = target.with_name(target.name + ".rebuilding")
            shutil.copyfile(built[destination], staged[destination])
        for destination in changed:
            staged[destination].replace(root / destination)
            del staged[destination]
    finally:
        for left in staged.values():
            left.unlink(missing_ok=True)
    return changed


def _reasons(missing: dict[str, str]) -> str:
    return "; ".join(f"{destination} ({reason})" for destination, reason in sorted(missing.items()))


def check(rebuild: Rebuild, root: Path, log: Callable[[str], None] = print) -> int:
    """Report how the committed files compare with the rebuild."""
    stale = differing(rebuild.built, root)
    for destination in sorted({*rebuild.built, *rebuild.missing}):
        state = ("UNCHECKED" if destination in rebuild.missing
                 else "DIFFERS" if destination in stale else "same")
        log(f"{state:<9} {destination}")
    if stale:
        log(f"CHECK FAILED: {len(stale)} of {len(rebuild.built)} derived files differ from a rebuild. "
            "Run scripts/rebuild_derived.py to replace them.")
        return DIFFERENT
    if rebuild.missing:
        log(f"CHECK INCOMPLETE: {len(rebuild.built)} derived files equal a rebuild, and "
            f"{len(rebuild.missing)} could not be rebuilt: {_reasons(rebuild.missing)}.")
        return INCOMPLETE
    log(f"CHECK OK: {len(rebuild.built)} derived files equal a rebuild")
    return 0


def replace(rebuild: Rebuild, root: Path, log: Callable[[str], None] = print) -> int:
    """Install the rebuild, or nothing where a file could not be rebuilt."""
    if rebuild.missing:
        log(f"NOTHING REPLACED: {_reasons(rebuild.missing)}. Restore {SUBMITTED_BOARD} "
            "and run this script again.")
        return INCOMPLETE
    changed = install(rebuild.built, root)
    for destination in changed:
        log(f"replaced {destination}")
    log(f"{len(changed)} of {len(rebuild.built)} derived files changed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="Compare the committed files with a rebuild and write nothing.")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="pcpbench_rebuild_") as tmp:
        try:
            rebuild = run_steps(Path(tmp), ROOT)
        except (RuntimeError, OSError) as error:
            print(f"REBUILD FAILED: {error}", file=sys.stderr)
            return INCOMPLETE
        return check(rebuild, ROOT) if args.check else replace(rebuild, ROOT)


if __name__ == "__main__":
    sys.exit(main())
