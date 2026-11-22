"""The ``score`` option of the six LAC-family calibrators, the runner and the grid script: APS
runs of split, mondrian, weighted, ppi, oracle and clustered on the same caches, under separately
named rows."""

import os
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pcpbench.calibrators.clustered import ClusteredMondrian
from pcpbench.calibrators.mondrian import SourceMondrian
from pcpbench.calibrators.oracle import OracleTargetMondrian
from pcpbench.calibrators.ppi import PPIMondrian
from pcpbench.calibrators.quantile import split_quantile
from pcpbench.calibrators.split import SplitCP
from pcpbench.calibrators.weighted import WeightedCP
from pcpbench.cli import main
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench import runner
from pcpbench.runner import run
from pcpbench.scores.matrix import SCORES, score_matrix
from pcpbench.scores.softmax_scores import lac_scores, softmax

REPO = Path(__file__).resolve().parents[1]
SIX = ("split", "mondrian", "weighted", "ppi", "oracle", "clustered")
CLASSES = {
    "split": SplitCP, "mondrian": SourceMondrian, "weighted": WeightedCP,
    "ppi": PPIMondrian, "oracle": OracleTargetMondrian, "clustered": ClusteredMondrian,
}


def _fit_kwargs(ds, method):
    tl, ty = ds.target_test()
    kw = {"alpha": 0.1, "seed": 0}
    if method in ("weighted", "ppi"):
        kw["target_logits"] = tl
    if method in ("ppi", "oracle"):
        kw["target_labeled"] = (tl[::2], ty[::2])
    return kw


@pytest.mark.parametrize("method", SIX)
def test_six_methods_fit_and_predict_under_aps(method):
    ds = make_synthetic_shift(K=6, n_per_class=200, seed=1)
    x, y = ds.source_cal()
    tl, _ = ds.target_test()
    fitted = CLASSES[method]().fit(x, y, score="aps", **_fit_kwargs(ds, method))
    mask = fitted.predict_set(tl)
    assert mask.shape == tl.shape and mask.dtype == bool
    thr = np.asarray(fitted.thresholds())
    assert np.all(np.isfinite(thr))
    if method != "weighted":
        assert np.array_equal(mask, score_matrix(tl, "aps") <= thr)


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("method", ("split", "mondrian", "clustered"))
def test_exchangeable_marginal_coverage_under_both_scores(method, score):
    ds = make_synthetic_shift(K=6, n_per_class=600, seed=2)
    x, y = ds.source_cal()
    perm = np.random.default_rng(0).permutation(len(y))
    a, b = perm[: len(y) // 2], perm[len(y) // 2:]
    fitted = CLASSES[method]().fit(x[a], y[a], alpha=0.1, seed=0, score=score)
    mask = fitted.predict_set(x[b])
    cov = float(mask[np.arange(len(b)), y[b]].mean())
    assert 0.85 <= cov <= 0.95, (method, score, cov)


@pytest.mark.parametrize("method", SIX)
def test_unknown_score_raises_in_fit(method):
    ds = make_synthetic_shift(K=4, n_per_class=60, seed=0)
    x, y = ds.source_cal()
    with pytest.raises(ValueError):
        CLASSES[method]().fit(x, y, score="raps", **_fit_kwargs(ds, method))


def test_split_thresholds_identical_with_and_without_explicit_lac():
    ds = make_synthetic_shift(seed=0)
    x, y = ds.source_cal()
    t_default = SplitCP().fit(x, y, alpha=0.1).thresholds()
    t_lac = SplitCP().fit(x, y, alpha=0.1, score="lac").thresholds()
    assert np.array_equal(t_default, t_lac)
    ref = split_quantile(lac_scores(softmax(x.astype(np.float64)), y), 0.1)
    assert float(t_default) == ref


@pytest.mark.parametrize("method", SIX)
def test_lac_default_equals_explicit_lac_for_every_method(method):
    ds = make_synthetic_shift(K=6, n_per_class=150, seed=3)
    x, y = ds.source_cal()
    tl, _ = ds.target_test()
    kw = _fit_kwargs(ds, method)
    f0 = CLASSES[method]().fit(x, y, **kw)
    f1 = CLASSES[method]().fit(x, y, score="lac", **kw)
    assert np.array_equal(np.asarray(f0.thresholds()), np.asarray(f1.thresholds()))
    assert np.array_equal(f0.predict_set(tl), f1.predict_set(tl))


def test_split_and_single_cluster_clustered_agree_on_float32_logits():
    """Every method scores in float64: SplitCP and ClusteredMondrian with no clustering points
    (so every class joins the null cluster and the whole set calibrates) take the same threshold
    on the same float32 logits."""
    rng = np.random.default_rng(11)
    K, n = 10, 4000
    y = rng.integers(0, K, size=n)
    z = rng.normal(0.0, 1.0, size=(n, K))
    z[np.arange(n), y] += 2.0
    z = z.astype(np.float32)
    t_split = float(SplitCP().fit(z, y, alpha=0.1).thresholds())
    fitted = ClusteredMondrian(n_clustering=0).fit(z, y, alpha=0.1, seed=0)
    assert fitted.clustering_skipped and fitted.calibration_mask.all()
    assert np.max(np.abs(fitted.thresholds() - t_split)) <= 1e-12


def test_runner_names_aps_rows_and_records_score():
    ds = make_synthetic_shift(K=6, n_per_class=150, seed=0)
    row = run(ds, "split", alpha=0.1, seeds=2, score="aps")
    assert row["method"] == "split_aps" and row["score"] == "aps"
    row0 = run(ds, "split", alpha=0.1, seeds=2)
    assert row0["method"] == "split" and row0["score"] == "lac"


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("method", SIX)
def test_runner_records_score_and_set_rule_of_the_six(method, score):
    ds = make_synthetic_shift(K=4, n_per_class=80, seed=0)
    row = run(ds, method, alpha=0.1, seeds=1, score=score)
    assert row["score"] == score
    assert row["set_rule"] == "score_threshold"


@pytest.mark.parametrize("score", SCORES)
def test_runner_records_audited_aps_and_greedy_rule_for_acp(score):
    """acp scores with its own audited APS under every option, and its set keeps the class that
    crosses the threshold."""
    ds = make_synthetic_shift(K=4, n_per_class=80, seed=0)
    row = run(ds, "acp", alpha=0.1, seeds=1, score=score)
    assert row["method"] == "acp"
    assert row["score"] == "audited_aps"
    assert row["set_rule"] == "greedy_crossing"


def test_score_threshold_rows_can_hold_empty_sets():
    """The exact set keeps every class whose score is at most the threshold, so under APS a row's
    set can be empty (a miss)."""
    ds = make_synthetic_shift(seed=0)
    x, y = ds.source_cal()
    tl, _ = ds.target_test()
    mask = SplitCP().fit(x, y, alpha=0.1, score="aps").predict_set(tl)
    assert (mask.sum(axis=1) == 0).any()


def test_runner_rejects_unknown_score():
    ds = make_synthetic_shift(K=4, n_per_class=60, seed=0)
    with pytest.raises(ValueError):
        run(ds, "split", alpha=0.1, seeds=1, score="raps")


def test_runner_passes_score_to_six_methods(monkeypatch):
    ds = make_synthetic_shift(K=4, n_per_class=80, seed=0)
    seen = {}
    for m in SIX:
        real = runner.METHODS[m]

        def make(real=real, m=m):
            obj = real()
            orig = obj.fit

            def fit(*a, **kw):
                seen[m] = kw.get("score")
                return orig(*a, **kw)
            obj.fit = fit
            return obj
        monkeypatch.setitem(runner.METHODS, m, make)
    for m in SIX:
        run(ds, m, alpha=0.1, seeds=1, score="aps")
    assert seen == {m: "aps" for m in SIX}


def test_acp_never_receives_score_and_runs_unchanged(monkeypatch):
    ds = make_synthetic_shift(K=4, n_per_class=80, seed=0)
    base = run(ds, "acp", alpha=0.1, seeds=2)
    real = runner.METHODS["acp"]

    def make():
        obj = real()
        orig = obj.fit

        def fit(*a, **kw):
            assert "score" not in kw
            return orig(*a, **kw)
        obj.fit = fit
        return obj
    monkeypatch.setitem(runner.METHODS, "acp", make)
    row = run(ds, "acp", alpha=0.1, seeds=2, score="aps")
    assert row["method"] == "acp"
    for key in ("marginal", "worst_class", "frac_undercovered", "mean_set_size",
                "frac_sig_undercovered", "per_class_coverage_per_seed"):
        assert row[key] == base[key]


def test_cli_score_option(tmp_path):
    import json
    out = tmp_path / "row.json"
    assert main(["run", "--dataset", "synthetic", "--method", "mondrian",
                 "--score", "aps", "--seeds", "1", "--out", str(out)]) == 0
    row = json.loads(out.read_text())
    assert row["method"] == "mondrian_aps" and row["score"] == "aps"
    with pytest.raises(SystemExit):
        main(["run", "--dataset", "synthetic", "--method", "split", "--score", "raps"])


def test_grid_scripts_parse():
    for script in ("run_grid.sh", "run_grid.sbatch", "run_aps_grid.sbatch"):
        r = subprocess.run(["bash", "-n", str(REPO / "scripts" / script)],
                           capture_output=True, text=True)
        assert r.returncode == 0, (script, r.stderr)


def _fake_py(tmp_path: Path) -> Path:
    log = tmp_path / "calls.txt"
    fake = tmp_path / "fakepy"
    fake.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log}"\necho "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _dry_grid(tmp_path: Path, **env_extra) -> list[str]:
    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "toy.npz").write_bytes(b"")
    env = dict(os.environ, PY=str(_fake_py(tmp_path)), CACHES=str(caches), **env_extra)
    r = subprocess.run(["bash", str(REPO / "scripts/run_grid.sh")], cwd=tmp_path,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return (tmp_path / "calls.txt").read_text().splitlines()


SEVEN = ["split", "mondrian", "weighted", "ppi", "acp", "oracle", "clustered"]


def test_grid_dry_run_under_aps(tmp_path):
    all_calls = _dry_grid(tmp_path, SCORE="aps")
    calls = [c for c in all_calls if "pcpbench.cli" in c]
    methods = [c.split("--method ")[1].split()[0] for c in calls]
    assert methods == list(SIX)
    for c, m in zip(calls, methods):
        assert "--score aps" in c
        assert f"--out outputs/aps/toy_{m}_aps.json" in c
    board = [c for c in all_calls if "pcpbench.leaderboard" in c]
    assert board == ["-m pcpbench.leaderboard --in outputs/aps --out leaderboard_aps.md"]
    assert (tmp_path / "outputs" / "aps").is_dir()


def test_grid_dry_run_default_keeps_names(tmp_path):
    all_calls = _dry_grid(tmp_path)
    calls = [c for c in all_calls if "pcpbench.cli" in c]
    methods = [c.split("--method ")[1].split()[0] for c in calls]
    assert methods == SEVEN
    for c, m in zip(calls, methods):
        assert "--score lac" in c
        assert f"--out outputs/toy_{m}.json" in c
    board = [c for c in all_calls if "pcpbench.leaderboard" in c]
    assert board == ["-m pcpbench.leaderboard --in outputs --out leaderboard.md"]
    assert not (tmp_path / "outputs" / "aps").exists()


def _dry_sbatch(tmp_path: Path, script: str) -> tuple[list[str], str]:
    text = (REPO / "scripts" / script).read_text()
    kept = [ln for ln in text.splitlines()
            if not ln.startswith(("source ", "conda ", "cd ", "export PY="))]
    copy = tmp_path / script
    copy.write_text("\n".join(kept) + "\n")
    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "toy.npz").write_bytes(b"")
    env = dict(os.environ, PY=str(_fake_py(tmp_path)))
    r = subprocess.run(["bash", str(copy)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return (tmp_path / "calls.txt").read_text().splitlines(), text


def test_aps_sbatch_writes_to_its_own_directory(tmp_path):
    calls, text = _dry_sbatch(tmp_path, "run_aps_grid.sbatch")
    cells = [c for c in calls if "pcpbench.cli" in c]
    methods = sorted(c.split("--method ")[1].split()[0] for c in cells)
    assert methods == sorted(SIX)
    for c in cells:
        m = c.split("--method ")[1].split()[0]
        assert "--score aps" in c
        assert f"--out outputs/aps/toy_{m}_aps.json" in c
    board = [c for c in calls if "pcpbench.leaderboard" in c]
    assert board == ["-m pcpbench.leaderboard --in outputs/aps --out leaderboard_aps.md"]
    assert "both sets of rows" not in text


def test_lac_sbatch_runs_every_registry_method(tmp_path):
    calls, _ = _dry_sbatch(tmp_path, "run_grid.sbatch")
    cells = [c for c in calls if "pcpbench.cli" in c]
    assert sorted(c.split("--method ")[1].split()[0] for c in cells) == sorted(runner.METHODS)
    for c in cells:
        m = c.split("--method ")[1].split()[0]
        assert f"--out outputs/toy_{m}.json" in c
    board = [c for c in calls if "pcpbench.leaderboard" in c]
    assert board == ["-m pcpbench.leaderboard --in outputs --out leaderboard.md"]


@pytest.mark.parametrize("script", ["run_grid.sh", "run_grid.sbatch", "run_aps_grid.sbatch"])
def test_a_grid_without_a_cache_fails_before_the_board(tmp_path, script):
    """A grid script with no cache to run fails and leaves the board as it is."""
    text = (REPO / "scripts" / script).read_text()
    kept = [ln for ln in text.splitlines()
            if not ln.startswith(("source ", "conda ", "cd ", "export PY="))]
    copy = tmp_path / script
    copy.write_text("\n".join(kept) + "\n")
    (tmp_path / "caches").mkdir()
    env = dict(os.environ, PY=str(_fake_py(tmp_path)), CACHES=str(tmp_path / "caches"))
    r = subprocess.run(["bash", str(copy)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "No caches found" in r.stdout
    calls = tmp_path / "calls.txt"
    assert not calls.exists() or "pcpbench" not in calls.read_text()


def test_run_grid_runs_one_blas_thread_unless_the_caller_sets_a_count(tmp_path):
    """run_grid.sh runs every cell on one BLAS thread, as the released rows ran."""
    caches = tmp_path / "caches"
    caches.mkdir()
    (caches / "toy.npz").write_bytes(b"")
    seen = tmp_path / "threads.txt"
    fake = tmp_path / "fakepy"
    fake.write_text('#!/usr/bin/env bash\n'
                    f'echo "$OMP_NUM_THREADS $MKL_NUM_THREADS $OPENBLAS_NUM_THREADS" >> "{seen}"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    base = {k: v for k, v in os.environ.items()
            if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for extra, expected in (({}, "1 1 1"), ({"OMP_NUM_THREADS": "4"}, "4 1 1")):
        seen.unlink(missing_ok=True)
        env = dict(base, PY=str(fake), CACHES=str(caches), METHODS="split", **extra)
        r = subprocess.run(["bash", str(REPO / "scripts/run_grid.sh")], cwd=tmp_path,
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert set(seen.read_text().splitlines()) == {expected}


def _row_pair(tmp_path: Path, dataset: str) -> tuple[dict, dict]:
    ds = make_synthetic_shift(K=6, n_per_class=150, seed=0)
    lac = run(ds, "split", alpha=0.1, seeds=1)
    aps = run(ds, "split", alpha=0.1, seeds=1, score="aps")
    lac["dataset"] = aps["dataset"] = dataset
    return lac, aps


def test_leaderboard_does_not_descend_into_the_aps_directory(tmp_path):
    import json
    from pcpbench.leaderboard import main as board_main
    lac, aps = _row_pair(tmp_path, "synthetic_shift")
    (tmp_path / "aps").mkdir()
    (tmp_path / "synthetic_split.json").write_text(json.dumps(lac))
    (tmp_path / "aps" / "synthetic_split_aps.json").write_text(json.dumps(aps))
    out = tmp_path / "leaderboard.md"
    assert board_main(["--in", str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text()
    assert "split_aps" not in text
    assert "| split |" in text


def _load_analyze():
    import importlib.util
    import sys
    path = REPO / "scripts" / "analyze_leaderboard.py"
    spec = importlib.util.spec_from_file_location("analyze_leaderboard_t4b", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_headline_drops_rows_of_a_non_lac_score(tmp_path):
    import json
    from pcpbench.leaderboard import main as board_main
    al = _load_analyze()
    assert tuple(al.SCORES) == SCORES
    lac, aps = _row_pair(tmp_path, "cifar10c_brightness_s1")
    d = tmp_path / "rows"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(lac))
    (d / "b.json").write_text(json.dumps(aps))
    clustered = dict(lac, method="clustered")
    (d / "c.json").write_text(json.dumps(clustered))
    out = tmp_path / "leaderboard.md"
    assert board_main(["--in", str(d), "--out", str(out)]) == 0
    rows = al.parse_leaderboard(out.read_text())
    assert sorted(r.method for r in rows) == ["clustered", "split", "split_aps"]
    assert [r.method for r in al.select_headline(rows)] == ["split"]
    assert al.HEADLINE_METHODS == frozenset({"split", "mondrian", "weighted", "ppi", "acp", "oracle"})


def test_grid_dry_run_methods_env(tmp_path):
    calls = [c for c in _dry_grid(tmp_path, METHODS="split ppi")
             if "pcpbench.cli" in c]
    assert [c.split("--method ")[1].split()[0] for c in calls] == ["split", "ppi"]
