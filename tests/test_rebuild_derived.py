"""Gates on scripts/rebuild_derived.py: the derived artifacts as committed are the ones their
commands produce from the committed row files and sweeps."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "rebuild_derived.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rebuild_derived", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rebuild() -> ModuleType:
    return _load()


def test_the_committed_derived_files_equal_a_fresh_rebuild(rebuild: ModuleType) -> None:
    done = subprocess.run([sys.executable, str(_SCRIPT), "--check"], cwd=_ROOT,
                          capture_output=True, text=True)
    report = done.stdout + done.stderr
    assert "DIFFERS" not in done.stdout and "CHECK FAILED" not in done.stdout, report
    if rebuild.submitted_board(_ROOT) is None:
        assert done.returncode == rebuild.INCOMPLETE and "CHECK INCOMPLETE" in done.stdout, report
        pytest.skip(f"{rebuild.SUBMITTED_BOARD} is missing, so the cell-level difference was "
                    "not checked; the other derived files equal a rebuild")
    assert done.returncode == 0, report
    n_files = sum(len(step.writes) for step in rebuild.steps())
    assert f"CHECK OK: {n_files} derived files equal a rebuild" in done.stdout, report


def test_a_changed_or_missing_file_is_reported_and_replaced(rebuild: ModuleType, tmp_path: Path) -> None:
    root, work = tmp_path / "repo", tmp_path / "work"
    (root / "a").mkdir(parents=True)
    work.mkdir()
    (root / "a" / "same.json").write_text("1\n")
    (root / "a" / "stale.json").write_text("old\n")
    for name, text in (("same.json", "1\n"), ("stale.json", "new\n"), ("fresh.md", "x\n")):
        (work / name).write_text(text)
    built = {"a/same.json": work / "same.json", "a/stale.json": work / "stale.json",
             "b/fresh.md": work / "fresh.md"}
    assert rebuild.differing(built, root) == ["a/stale.json", "b/fresh.md"]
    assert rebuild.install(built, root) == ["a/stale.json", "b/fresh.md"]
    assert (root / "a" / "stale.json").read_text() == "new\n"
    assert (root / "b" / "fresh.md").read_text() == "x\n"
    assert rebuild.differing(built, root) == []
    assert not list(root.rglob("*.rebuilding"))


def test_the_steps_run_in_the_order_they_depend_on_each_other(rebuild: ModuleType) -> None:
    """The board comes first, since three later steps read it."""
    order = [next(iter(step.writes)) for step in rebuild.steps()]
    assert order[0] == rebuild.BOARD
    readers = [step for step in rebuild.steps() if rebuild.BOARD in step.command[1:]
               and rebuild.BOARD not in step.writes]
    assert len(readers) == 3


@pytest.mark.parametrize("old, new, same", [
    (0.7651879999999999, 0.765188, True),
    (0.1, math.nextafter(0.1, 1.0), True),
    (0.0, 2.220446049250313e-16, True),
    (2.917, 2.9172, False),
    (0.718, 0.7180001, False),
    (float("nan"), float("nan"), True),
    (float("inf"), float("inf"), True),
    (float("inf"), 1e308, False),
    (1, 1.0, False),
    (True, 1, False),
    ("0.5", 0.5, False),
    (None, None, True),
    (None, 0, False),
    ({"a": [1, {"b": 0.1}]}, {"a": [1, {"b": math.nextafter(0.1, 1.0)}]}, True),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
    ({"a": 1}, {"a": 1, "b": 2}, False),
    ([1, 2], [1, 2, 3], False),
    ([0.5, 1], [1, 0.5], False),
])
def test_two_json_values_are_the_same_apart_from_the_last_digits_of_a_float(
        rebuild: ModuleType, old: object, new: object, same: bool) -> None:
    assert rebuild.same_json(old, new) is same
    assert rebuild.same_json(new, old) is same


def test_a_file_that_differs_in_the_last_digits_of_a_float_only_is_left_alone(
        rebuild: ModuleType, tmp_path: Path) -> None:
    """Under numpy 2.4.6 the rebuild of two committed files differs from the files in the last
    digits of 75 floats (measured 2026-11-10), and nothing in them had changed."""
    root, work = tmp_path / "repo", tmp_path / "work"
    (root / "a").mkdir(parents=True)
    work.mkdir()
    committed = {"noise.json": '{"mean": 0.7651879999999999, "n": 400}\n',
                 "moved.json": '{"mean": 0.765188, "n": 400}\n',
                 "count.json": '{"mean": 0.765188, "n": 400}\n',
                 "broken.json": "{not json\n",
                 "NOTE.md": "mean 0.7651879999999999\n"}
    rebuilt = {"noise.json": '{"mean": 0.765188, "n": 400}\n',
               "moved.json": '{"mean": 0.765189, "n": 400}\n',
               "count.json": '{"mean": 0.765188, "n": 401}\n',
               "broken.json": '{"mean": 0.5}\n',
               "NOTE.md": "mean 0.765188\n"}
    for name in committed:
        (root / "a" / name).write_text(committed[name])
        (work / name).write_text(rebuilt[name])
    built = {f"a/{name}": work / name for name in committed}
    changed = ["a/NOTE.md", "a/broken.json", "a/count.json", "a/moved.json"]
    assert rebuild.differing(built, root) == changed
    assert rebuild.install(built, root) == changed
    assert (root / "a" / "noise.json").read_text() == committed["noise.json"]
    assert (root / "a" / "moved.json").read_text() == rebuilt["moved.json"]
    assert rebuild.differing(built, root) == []


def _two_files(rebuild: ModuleType, tmp_path: Path, *, stale: bool, missing: bool) -> tuple[Path, Any]:
    root, work = tmp_path / "repo", tmp_path / "work"
    (root / "a").mkdir(parents=True)
    work.mkdir()
    for name, old, new in (("kept.json", "1\n", "1\n"), ("other.json", "1\n", "2\n" if stale else "1\n"),
                           ("diff.json", "old\n", None)):
        (root / "a" / name).write_text(old)
        if new is not None:
            (work / name).write_text(new)
    reason = f"{rebuild.SUBMITTED_BOARD} is missing"
    return root, rebuild.Rebuild(
        built={"a/kept.json": work / "kept.json", "a/other.json": work / "other.json"},
        missing={"a/diff.json": reason} if missing else {})


def test_the_check_says_which_file_it_could_not_rebuild(rebuild: ModuleType, tmp_path: Path) -> None:
    """A checkout without the submitted board must not end in CHECK OK with one file fewer: the
    registry reads that file."""
    root, result = _two_files(rebuild, tmp_path, stale=False, missing=True)
    lines: list[str] = []
    assert rebuild.check(result, root, log=lines.append) == rebuild.INCOMPLETE == 2
    assert lines[:3] == ["UNCHECKED a/diff.json", "same      a/kept.json", "same      a/other.json"]
    assert lines[3].startswith("CHECK INCOMPLETE: 2 derived files equal a rebuild, and 1 could not be rebuilt: "
                               "a/diff.json (outputs/analysis/leaderboard_submitted.md is missing)")
    assert not [line for line in lines if "CHECK OK" in line]


def test_the_check_reports_a_difference_before_a_file_it_could_not_rebuild(
        rebuild: ModuleType, tmp_path: Path) -> None:
    root, result = _two_files(rebuild, tmp_path, stale=True, missing=True)
    lines: list[str] = []
    assert rebuild.check(result, root, log=lines.append) == rebuild.DIFFERENT == 1
    assert "DIFFERS   a/other.json" in lines and "UNCHECKED a/diff.json" in lines
    assert lines[-1].startswith("CHECK FAILED: 1 of 2 derived files differ from a rebuild.")


def test_the_check_is_ok_only_with_every_file_rebuilt_and_equal(rebuild: ModuleType, tmp_path: Path) -> None:
    root, result = _two_files(rebuild, tmp_path, stale=False, missing=False)
    lines: list[str] = []
    assert rebuild.check(result, root, log=lines.append) == 0
    assert lines[-1] == "CHECK OK: 2 derived files equal a rebuild"


def test_nothing_is_replaced_while_a_file_cannot_be_rebuilt(rebuild: ModuleType, tmp_path: Path) -> None:
    """The board and its difference from the submitted board have to agree: the registry restores
    the submitted board from the pair."""
    root, result = _two_files(rebuild, tmp_path, stale=True, missing=True)
    lines: list[str] = []
    assert rebuild.replace(result, root, log=lines.append) == rebuild.INCOMPLETE
    assert (root / "a" / "other.json").read_text() == "1\n"
    assert len(lines) == 1 and lines[0].startswith("NOTHING REPLACED: a/diff.json (outputs/analysis/leaderboard_submitted.md ")
    whole = result._replace(missing={})
    assert rebuild.replace(whole, root, log=lines.append) == 0
    assert (root / "a" / "other.json").read_text() == "2\n"
    assert lines[1:] == ["replaced a/other.json", "1 of 2 derived files changed"]


def test_a_step_that_needs_the_submitted_board_is_left_out_where_that_board_is_missing(
        rebuild: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "outputs").mkdir(parents=True)
    (root / rebuild.BOARD).write_text("| dataset |\n")
    write = "import sys; open(sys.argv[1], 'w').write('1')"
    monkeypatch.setattr(rebuild, "steps", lambda: (
        rebuild.Step("a file of the row files", ("-c", write, "kept.json"), {"kept.json": "a/kept.json"}),
        rebuild.Step("a file of the submitted board", ("-c", write, "diff.json"), {"diff.json": "a/diff.json"},
                     needs_submitted_board=True)))
    reason = f"{rebuild.SUBMITTED_BOARD} is missing"

    monkeypatch.setattr(rebuild, "submitted_board", lambda _root: None)
    (tmp_path / "shallow").mkdir()
    lines: list[str] = []
    result = rebuild.run_steps(tmp_path / "shallow", root, log=lines.append)
    assert sorted(result.built) == ["a/kept.json"] and result.missing == {"a/diff.json": reason}
    assert not (tmp_path / "shallow" / "diff.json").exists()
    assert f"SKIPPED a file of the submitted board: {reason}" in lines

    monkeypatch.setattr(rebuild, "submitted_board", lambda _root: "| dataset |\n")
    (tmp_path / "full").mkdir()
    result = rebuild.run_steps(tmp_path / "full", root, log=lines.append)
    assert sorted(result.built) == ["a/diff.json", "a/kept.json"] and result.missing == {}


@pytest.mark.parametrize("arguments", [["--check"], []])
def test_a_step_that_stops_ends_the_run_as_not_rebuilt_and_not_as_a_difference(
        rebuild: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], arguments: list[str]) -> None:
    """Exit code 1 says that a committed file differs from its rebuild, and a step that stopped
    has shown no such thing."""
    root = tmp_path / "repo"
    (root / "outputs").mkdir(parents=True)
    (root / rebuild.BOARD).write_text("| dataset |\n")
    (root / "a").mkdir()
    (root / "a" / "kept.json").write_text("0\n")
    write = "import sys; open(sys.argv[1], 'w').write('1')"
    stop = "import sys; sys.exit('no row file for cifar100c')"
    monkeypatch.setattr(rebuild, "ROOT", root)
    monkeypatch.setattr(rebuild, "submitted_board", lambda _root: None)
    monkeypatch.setattr(rebuild, "steps", lambda: (
        rebuild.Step("a file of the row files", ("-c", write, "kept.json"), {"kept.json": "a/kept.json"}),
        rebuild.Step("the step that stops", ("-c", stop), {"late.json": "a/late.json"})))
    assert rebuild.main(arguments) == rebuild.INCOMPLETE
    assert "REBUILD FAILED: the step that stops failed:\nno row file for cifar100c" in capsys.readouterr().err
    assert (root / "a" / "kept.json").read_text() == "0\n"


def test_no_file_is_replaced_when_one_of_them_cannot_be_staged(
        rebuild: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file is copied beside its place before the first one is renamed into it, so a copy
    that fails (a full disk) leaves the committed files as they were."""
    root, work = tmp_path / "repo", tmp_path / "work"
    (root / "a").mkdir(parents=True)
    work.mkdir()
    built = {}
    for name in ("first.json", "second.json"):
        (root / "a" / name).write_text("0\n")
        (work / name).write_text("1\n")
        built[f"a/{name}"] = work / name
    copy = rebuild.shutil.copyfile

    def copy_until_the_disk_is_full(source: Path, target: Path) -> None:
        if Path(source).name == "second.json":
            raise OSError(28, "No space left on device")
        copy(source, target)

    monkeypatch.setattr(rebuild.shutil, "copyfile", copy_until_the_disk_is_full)
    with pytest.raises(OSError, match="No space left"):
        rebuild.install(built, root)
    assert [(root / "a" / name).read_text() for name in ("first.json", "second.json")] == ["0\n", "0\n"]
    assert not list(root.rglob("*.rebuilding"))
    monkeypatch.setattr(rebuild.shutil, "copyfile", copy)
    assert rebuild.install(built, root) == ["a/first.json", "a/second.json"]
    assert [(root / "a" / name).read_text() for name in ("first.json", "second.json")] == ["1\n", "1\n"]


def test_the_check_replaces_nothing_and_the_rebuild_does(
        rebuild: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "repo"
    (root / "outputs").mkdir(parents=True)
    (root / rebuild.BOARD).write_text("| dataset |\n")
    (root / "a").mkdir()
    (root / "a" / "kept.json").write_text("0\n")
    write = "import sys; open(sys.argv[1], 'w').write('1')"
    monkeypatch.setattr(rebuild, "ROOT", root)
    monkeypatch.setattr(rebuild, "submitted_board", lambda _root: None)
    monkeypatch.setattr(rebuild, "steps", lambda: (
        rebuild.Step("a file of the row files", ("-c", write, "kept.json"), {"kept.json": "a/kept.json"}),))
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    assert rebuild.main(["--check"]) == rebuild.DIFFERENT
    assert "DIFFERS   a/kept.json" in capsys.readouterr().out
    assert (root / "a" / "kept.json").read_text() == "0\n"
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == before

    assert rebuild.main([]) == 0
    assert "replaced a/kept.json" in capsys.readouterr().out
    assert (root / "a" / "kept.json").read_text() == "1"
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == before
