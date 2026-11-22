"""Tests for scripts/run_nested_ksweep.sh and its sbatch wrapper."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_SH = _SCRIPTS / "run_nested_ksweep.sh"
_SBATCH = _SCRIPTS / "run_nested_ksweep.sbatch"

_FAKE_PY = """#!/usr/bin/env bash
echo "FAKE_PY_CALLED $*"
case "$*" in
    *"${FAIL_MARK:-@@never@@}"*) exit 3 ;;
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

_SHIFT_CACHES = [
    "cifar100c_contrast_s3.npz",
    "cifar100c_fog_s1.npz",
    "cifar100c_vitL_contrast_s3.npz",
]
_DECOYS = [
    "cifar100c.npz",
    "cifar100c_mild.npz",
    "cifar100c_vitL.npz",
    "cifar10c_fog_s1.npz",
]
_OFF_GRID = [
    "cifar100c_impulse_noise_s1.npz",
    "cifar100c_fog_s3_old.npz",
    "cifar100c_vitL_fog_s6.npz",
]
_SHIFT_NAMES = ["contrast_s3", "fog_s1", "vitL_contrast_s3"]
_NOSHIFT_NAMES = [f"{b}_v{v}" for b in ("vitB", "vitL") for v in range(5)]


def _workdir(tmp_path: Path, caches=_SHIFT_CACHES + _DECOYS + _OFF_GRID) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(_SH, scripts / "run_nested_ksweep.sh")
    fake = tmp_path / "fake_py.sh"
    fake.write_text(_FAKE_PY)
    fake.chmod(0o755)
    cdir = tmp_path / "caches"
    cdir.mkdir()
    for name in caches:
        (cdir / name).write_bytes(b"")
    return tmp_path


def _run(work: Path, **extra):
    env = {
        "PY": str(work / "fake_py.sh"),
        "CACHES": str(work / "caches"),
        "PATH": "/usr/bin:/bin",
    }
    env.update({k: str(v) for k, v in extra.items()})
    return subprocess.run(
        ["bash", "scripts/run_nested_ksweep.sh"],
        cwd=work, env=env, capture_output=True, text=True,
    )


def _out_files(work: Path):
    return sorted(p.name for p in (work / "outputs" / "nested_ksweep").iterdir())


def _expected_outputs():
    return sorted(
        [f"nested_k_sweep_{n}.json.gz" for n in _SHIFT_NAMES]
        + [f"noshift_k_sweep_{n}.json.gz" for n in _NOSHIFT_NAMES]
    )


def _calls(text: str):
    return [ln for ln in text.splitlines() if ln.startswith("FAKE_PY_CALLED")]


@pytest.mark.parametrize("path", [_SH, _SBATCH], ids=lambda p: p.name)
def test_bash_n_passes(path):
    assert path.exists(), f"missing {path}"
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_writes_every_expected_output_and_ignores_decoys(tmp_path):
    work = _workdir(tmp_path)
    res = _run(work)
    assert res.returncode == 0, res.stdout + res.stderr
    assert _out_files(work) == _expected_outputs()
    assert res.stdout.rstrip().splitlines()[-1] == (
        "NESTED K-SWEEP COMPLETED: all 13 runs present."
    )
    ignored = sorted(ln.split()[1].rsplit("/", 1)[-1] for ln in res.stdout.splitlines()
                     if ln.startswith("IGNORE: "))
    assert ignored == sorted(_OFF_GRID), res.stdout
    for name in _OFF_GRID:
        assert name[: -len(".npz")] not in " ".join(_out_files(work))


def test_each_log_holds_one_call_with_the_paper_flags(tmp_path):
    work = _workdir(tmp_path)
    assert _run(work).returncode == 0
    outdir = Path("outputs") / "nested_ksweep"
    required = [
        "--subset-design nested", "--subset-seeds 30", "--k-values 10,25,50,100",
        "--n-cal-per-class 50", "--alpha 0.10", "--eval-frac 0.5", "--stamp ",
    ]
    for name in _SHIFT_NAMES:
        log = (work / "logs" / f"nested_ksweep_{name}.log").read_text()
        lines = _calls(log)
        assert len(lines) == 1, log
        for frag in required:
            assert frag in lines[0], (frag, lines[0])
        assert f"--out {outdir / f'nested_k_sweep_{name}.json.gz'}" in lines[0]
        assert f"--cache {work / 'caches' / f'cifar100c_{name}.npz'}" in lines[0]
        assert "--view" not in lines[0]
    for backbone, base in (("vitB", "contrast_s3"), ("vitL", "vitL_contrast_s3")):
        for v in range(5):
            log = (work / "logs" / f"nested_ksweep_noshift_{backbone}_v{v}.log").read_text()
            lines = _calls(log)
            assert len(lines) == 1, log
            for frag in required:
                assert frag in lines[0], (frag, lines[0])
            assert f"--view no_shift --view-seed {v}" in lines[0]
            assert f"--cache {work / 'caches' / f'cifar100c_{base}.npz'}" in lines[0]
            out = outdir / f"noshift_k_sweep_{backbone}_v{v}.json.gz"
            assert f"--out {out}" in lines[0]
            assert lines[0].index("--view no_shift") < lines[0].index("--out ")


def test_seeds_and_stamp_are_passed_through(tmp_path):
    work = _workdir(tmp_path)
    assert _run(work, SEEDS=7, STAMP="abc1234").returncode == 0
    for log in (work / "logs").glob("nested_ksweep_*.log"):
        line = _calls(log.read_text())[0]
        assert "--subset-seeds 7" in line
        assert "--stamp abc1234" in line


def test_rerun_skips_all_and_force_reruns_all(tmp_path):
    work = _workdir(tmp_path)
    assert _run(work).returncode == 0
    second = _run(work)
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout.count("SKIP:") == 13, second.stdout
    assert _calls(second.stdout) == []
    for log in (work / "logs").glob("nested_ksweep_*.log"):
        log.unlink()
    third = _run(work, FORCE=1)
    assert third.returncode == 0, third.stdout + third.stderr
    assert third.stdout.count("DONE:") == 13, third.stdout
    assert len(list((work / "logs").glob("nested_ksweep_*.log"))) == 13


@pytest.mark.parametrize("change", [{"STAMP": "bbb2222"}, {"SEEDS": 7}],
                         ids=["new code version", "new seed count"])
def test_output_of_another_version_or_seed_count_is_rerun(tmp_path, change):
    """An output is skipped only when the same code version and seed count wrote it, so a
    resubmission after a code fix cannot leave files of two versions."""
    work = _workdir(tmp_path)
    assert _run(work, STAMP="aaa1111").returncode == 0
    same = _run(work, STAMP="aaa1111")
    assert same.stdout.count("SKIP:") == 13 and "RERUN:" not in same.stdout, same.stdout
    again = _run(work, **{"STAMP": "aaa1111", **change})
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.count("RERUN:") == 13 and again.stdout.count("DONE:") == 13, again.stdout
    assert "SKIP:" not in again.stdout
    want = f"{change.get('STAMP', 'aaa1111')} seeds={change.get('SEEDS', 30)} sha256="
    marks = sorted((work / "logs").glob("nested_ksweep_*.written_by"))
    assert len(marks) == 13 and all(m.read_text().startswith(want) for m in marks)


def test_output_without_a_record_of_its_writer_is_rerun(tmp_path):
    work = _workdir(tmp_path)
    out = work / "outputs" / "nested_ksweep"
    out.mkdir(parents=True)
    (out / "nested_k_sweep_fog_s1.json.gz").write_text("left by an unrecorded run")
    res = _run(work)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "RERUN: fog_s1 (" in res.stdout and "mark: 'none'" in res.stdout
    assert (out / "nested_k_sweep_fog_s1.json.gz").read_text() == "{}\n"


def test_mark_names_the_sha256_of_the_output_it_describes(tmp_path):
    import hashlib
    work = _workdir(tmp_path)
    assert _run(work, STAMP="aaa1111").returncode == 0
    out = work / "outputs" / "nested_ksweep" / "nested_k_sweep_fog_s1.json.gz"
    mark = (work / "logs" / "nested_ksweep_fog_s1.written_by").read_text().strip()
    assert mark == f"aaa1111 seeds=30 sha256={hashlib.sha256(out.read_bytes()).hexdigest()}"


def test_output_replaced_by_an_older_one_is_rerun_though_its_mark_names_this_version(tmp_path):
    """The results of an older code version are put back (a checkout, a copy) while the marks,
    which live in logs/, still name the new version."""
    work = _workdir(tmp_path)
    assert _run(work, STAMP="bbb2222").returncode == 0
    out = work / "outputs" / "nested_ksweep"
    for name in ("nested_k_sweep_fog_s1.json.gz", "noshift_k_sweep_vitB_v0.json.gz"):
        (out / name).write_text("the output of the older code version\n")
    res = _run(work, STAMP="bbb2222")
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.count("RERUN:") == 2 and res.stdout.count("SKIP:") == 11, res.stdout
    assert res.stdout.count("DONE:") == 2
    assert (out / "nested_k_sweep_fog_s1.json.gz").read_text() == "{}\n"


def test_final_check_rejects_an_output_that_no_longer_matches_its_mark(tmp_path):
    work = _workdir(tmp_path)
    assert _run(work, STAMP="bbb2222").returncode == 0
    (work / "outputs" / "nested_ksweep" / "nested_k_sweep_fog_s1.json.gz").write_text("old\n")
    res = _run(work, STAMP="bbb2222", FAIL_MARK="fog_s1")
    assert res.returncode == 1 and "  - fog_s1" in res.stdout, res.stdout


def test_stale_output_whose_rerun_fails_counts_as_failed(tmp_path):
    work = _workdir(tmp_path)
    assert _run(work, STAMP="aaa1111").returncode == 0
    res = _run(work, STAMP="bbb2222", FAIL_MARK="fog_s1")
    assert res.returncode == 1, res.stdout + res.stderr
    assert "NESTED K-SWEEP COMPLETED WITH 1 FAILED RUN(S):" in res.stdout
    assert "  - fog_s1" in res.stdout


def test_interpreter_must_be_given(tmp_path):
    work = _workdir(tmp_path)
    res = subprocess.run(["bash", "scripts/run_nested_ksweep.sh"], cwd=work,
                         env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert res.returncode != 0
    assert "set PY to the absolute path of the interpreter" in res.stderr


def test_failing_cache_is_reported_and_the_rest_complete(tmp_path):
    work = _workdir(tmp_path)
    res = _run(work, FAIL_MARK="fog_s1")
    assert res.returncode == 1, res.stdout + res.stderr
    assert re.search(r"^FAIL: fog_s1 ", res.stdout, re.M), res.stdout
    assert "NESTED K-SWEEP COMPLETED WITH 1 FAILED RUN(S):" in res.stdout
    assert "  - fog_s1" in res.stdout
    assert len(_out_files(work)) == 12
    assert "nested_k_sweep_fog_s1.json.gz" not in _out_files(work)


def test_missing_noshift_base_cache_fails_its_five_runs(tmp_path):
    caches = [c for c in _SHIFT_CACHES if c != "cifar100c_vitL_contrast_s3.npz"]
    caches.append("cifar100c_vitL_fog_s1.npz")
    work = _workdir(tmp_path, caches)
    res = _run(work)
    assert res.returncode == 1, res.stdout + res.stderr
    fails = [ln for ln in res.stdout.splitlines() if ln.startswith("FAIL: noshift_vitL_v")]
    assert len(fails) == 5, res.stdout
    assert all("not found" in ln for ln in fails)
    assert "NESTED K-SWEEP COMPLETED WITH 5 FAILED RUN(S):" in res.stdout


def test_no_shift_caches_is_a_failure(tmp_path):
    work = _workdir(tmp_path, _DECOYS + _OFF_GRID)
    res = _run(work)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "No grid caches matching" in res.stdout
    assert _calls(res.stdout) == []


def test_parallel_jobs_give_the_same_outputs_as_serial(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    serial = _workdir(tmp_path / "a")
    parallel = _workdir(tmp_path / "b")
    r1 = _run(serial, JOBS=1)
    r4 = _run(parallel, JOBS=4)
    assert r1.returncode == 0 and r4.returncode == 0, r4.stdout + r4.stderr
    assert _out_files(serial) == _out_files(parallel) == _expected_outputs()
    assert "all 13 runs present" in r4.stdout


@pytest.mark.parametrize("path", [_SH, _SBATCH], ids=lambda p: p.name)
def test_no_dashes_no_hidden_stderr_no_bare_python(path):
    text = path.read_text()
    assert "—" not in text and "–" not in text
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        if '"$PY"' in ln:
            assert "2>/dev/null" not in ln, ln
        assert not re.search(r"(^|[\s;&|(])python\s", ln), ln
    assert "2>/dev/null" not in "\n".join(
        ln for ln in text.splitlines() if "controlled_k_sweep" in ln
    )
