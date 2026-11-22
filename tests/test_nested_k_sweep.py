"""Tests for the nested class-subset design and the no-shift view of
scripts/controlled_k_sweep.py."""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pcpbench.datasets.shift_dataset import SOURCE_CAL, TARGET_TEST
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.validity_floor import expected_min_coverage

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "controlled_k_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("controlled_k_sweep_nested", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cks = _load_module()


def _with_row_ids(ds):
    return dataclasses.replace(ds, domain=np.arange(len(ds.labels), dtype=np.int32))


def _rows_by_class(sub, subset, seed, design):
    k = len(subset)
    class_ids = subset if design == "nested" else None
    tgt_logits, tgt_labels = sub.target_test()
    cal_idx, eval_idx = cks.split_target_labeled_and_eval(
        tgt_logits, tgt_labels, k, seed, 0.5, class_ids=class_ids)
    tgt_rows = sub.domain[sub.split == TARGET_TEST]
    src_mask = sub.split == SOURCE_CAL
    out = {}
    for c, g in enumerate(subset):
        out[int(g)] = {
            "source": frozenset(sub.domain[src_mask & (sub.labels == c)].tolist()),
            "pool": frozenset(tgt_rows[cal_idx][tgt_labels[cal_idx] == c].tolist()),
            "eval": frozenset(tgt_rows[eval_idx][tgt_labels[eval_idx] == c].tolist()),
        }
    return out


def test_nested_subsets_are_prefixes_of_one_class_order():
    for seed in range(10):
        order = cks.nested_class_order(20, seed)
        assert sorted(order.tolist()) == list(range(20))
        previous: set[int] = set()
        for k in (5, 10, 15, 20):
            subset = cks.draw_class_subset(k, 20, seed, design="nested")
            assert subset.tolist() == sorted(order[:k].tolist())
            assert previous < set(subset.tolist())
            previous = set(subset.tolist())
        assert previous == set(range(20))


def test_nested_class_order_varies_with_the_seed_and_repeats_for_one_seed():
    orders = [tuple(cks.nested_class_order(50, s).tolist()) for s in range(10)]
    assert len(set(orders)) == 10
    assert tuple(cks.nested_class_order(50, 3).tolist()) == orders[3]


def test_the_independent_draw_is_the_released_one():
    for seed in range(5):
        released = np.sort(np.random.default_rng(seed).choice(20, size=6, replace=False))
        assert cks.draw_class_subset(6, 20, seed).tolist() == released.tolist()
        assert cks.draw_class_subset(6, 20, seed, design="independent").tolist() == released.tolist()


def test_an_unknown_design_is_rejected():
    with pytest.raises(ValueError, match="design"):
        cks.draw_class_subset(5, 20, 0, design="paired")


def test_nested_draw_rejects_k_above_the_class_count():
    with pytest.raises(ValueError, match="exceeds"):
        cks.draw_class_subset(21, 20, 0, design="nested")


def test_nested_design_holds_every_class_rows_fixed_across_k():
    ds = _with_row_ids(make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0))
    for seed in range(4):
        rows = {}
        for k in (5, 10, 20):
            sub, shortfalls = cks.build_k_subproblem(ds, k, seed, 50, design="nested")
            assert shortfalls == {}
            subset = np.asarray(sub.meta["class_subset"])
            rows[k] = _rows_by_class(sub, subset, seed, "nested")
            for parts in rows[k].values():
                assert len(parts["source"]) == 50
                assert len(parts["pool"]) == 40 and len(parts["eval"]) == 40
                assert not parts["pool"] & parts["eval"]
        for small, large in ((5, 10), (5, 20), (10, 20)):
            assert set(rows[small]) < set(rows[large])
            for g, parts in rows[small].items():
                assert parts == rows[large][g], (seed, small, large, g)


def test_the_independent_design_does_not_hold_rows_fixed():
    """Under the independent design of the controlled sweep a class's rows change with K."""
    ds = _with_row_ids(make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0))
    moved = 0
    for seed in range(4):
        small, _ = cks.build_k_subproblem(ds, 5, seed, 50)
        large, _ = cks.build_k_subproblem(ds, 20, seed, 50)
        rows_small = _rows_by_class(small, np.asarray(small.meta["class_subset"]), seed, "independent")
        rows_large = _rows_by_class(large, np.asarray(large.meta["class_subset"]), seed, "independent")
        moved += sum(parts != rows_large[g] for g, parts in rows_small.items())
    assert moved > 0


def test_nested_rows_differ_between_seeds():
    ds = _with_row_ids(make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0))
    per_seed = []
    for seed in (0, 1):
        sub, _ = cks.build_k_subproblem(ds, 20, seed, 50, design="nested")
        per_seed.append(_rows_by_class(sub, np.asarray(sub.meta["class_subset"]), seed, "nested"))
    assert any(per_seed[0][g] != per_seed[1][g] for g in range(20))


def test_nested_shortfall_takes_every_row_and_is_reported():
    ds = make_synthetic_shift(K=6, n_per_class=30, shift=0.6, seed=0)
    sub, shortfalls = cks.build_k_subproblem(ds, 6, 0, 50, design="nested")
    assert shortfalls == {c: 20 for c in range(6)}
    assert int((sub.split == SOURCE_CAL).sum()) == 6 * 30


def test_every_seed_stream_is_distinct():
    """numpy joins a seed tuple into 32-bit words and pads with zeros, so (s, C, 0) and (s, C)
    are one stream and a value above 2**32 splits into two words."""
    tags = (cks.NESTED_ORDER_TAG, cks.NESTED_SOURCE_TAG, cks.NESTED_TARGET_TAG,
            cks.NO_SHIFT_VIEW_TAG)
    assert len(set(tags)) == len(tags)
    assert all(0 < tag < 2**32 for tag in tags)
    keys = []
    for s in range(4):
        keys += [(s,), (s, 7_000_000_000), (s, 2_000_000_000), (s, cks.NESTED_ORDER_TAG)]
        for g in range(100):
            keys += [(s, cks.NESTED_SOURCE_TAG, g + 1), (s, cks.NESTED_TARGET_TAG, g + 1),
                     (s, cks.NO_SHIFT_VIEW_TAG, g + 1)]
    states = [tuple(np.random.SeedSequence(key).generate_state(4).tolist()) for key in keys]
    assert len(set(states)) == len(keys)


def test_nested_sweep_runs_and_is_deterministic():
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    kwargs = dict(k_values=[5, 10, 20], subset_seeds=3, n_cal_per_class=50, alpha=0.10,
                  methods=["split", "mondrian", "oracle"], design="nested")
    first = cks.run_sweep(ds, **kwargs)
    second = cks.run_sweep(ds, **kwargs)
    assert first == second
    assert len(first) == 3 * 3 * 3
    for r in first:
        assert len(r.per_class_coverage) == r.k
        assert r.n_eval_total == 40 * r.k
        assert r.n_cal_total == 50 * r.k
        assert r.n_shortfall_classes == 0
        if r.method == "oracle":
            assert r.n_target_cal_pool == 40 * r.k


def test_the_default_sweep_ignores_the_new_argument():
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    kwargs = dict(k_values=[5, 10], subset_seeds=2, n_cal_per_class=50, alpha=0.10,
                  methods=["split", "mondrian", "oracle"])
    assert cks.run_sweep(ds, **kwargs) == cks.run_sweep(ds, design="independent", **kwargs)


def test_nested_and_independent_sweeps_differ():
    ds = make_synthetic_shift(K=20, n_per_class=80, shift=0.6, seed=0)
    kwargs = dict(k_values=[5], subset_seeds=3, n_cal_per_class=50, alpha=0.10,
                  methods=["mondrian"])
    assert cks.run_sweep(ds, **kwargs) != cks.run_sweep(ds, design="nested", **kwargs)


def test_no_shift_view_takes_its_target_from_source_rows_only():
    ds = _with_row_ids(make_synthetic_shift(K=8, n_per_class=200, shift=0.6, seed=0))
    source_rows = set(ds.domain[ds.split == SOURCE_CAL].tolist())
    view = cks.no_shift_view(ds, view_seed=0, n_target_per_class=60)
    assert view.n_classes == 8
    assert set(view.domain.tolist()) == source_rows
    assert len(set(view.domain.tolist())) == len(view.domain)
    for g in range(8):
        assert int(((view.split == TARGET_TEST) & (view.labels == g)).sum()) == 60
        assert int(((view.split == SOURCE_CAL) & (view.labels == g)).sum()) == 140
    assert np.array_equal(view.logits, ds.logits[view.domain])
    assert np.array_equal(view.labels, ds.labels[view.domain])
    assert view.meta["view"] == "no_shift" and view.meta["view_seed"] == 0


def test_no_shift_view_is_deterministic_and_varies_with_its_seed():
    ds = _with_row_ids(make_synthetic_shift(K=8, n_per_class=200, shift=0.6, seed=0))

    def target_rows(seed):
        view = cks.no_shift_view(ds, view_seed=seed, n_target_per_class=60)
        return frozenset(view.domain[view.split == TARGET_TEST].tolist())

    assert target_rows(0) == target_rows(0)
    assert target_rows(0) != target_rows(1)


def test_no_shift_view_rejects_a_class_with_too_few_source_rows():
    ds = make_synthetic_shift(K=8, n_per_class=100, shift=0.6, seed=0)
    with pytest.raises(ValueError, match="source"):
        cks.no_shift_view(ds, view_seed=0, n_target_per_class=100)


def test_per_class_methods_sit_on_the_reference_without_shift_and_fall_under_shift():
    """Source-Mondrian and the oracle are per-class valid on the no-shift view, so their worst
    class reads the matched reference."""
    ds = make_synthetic_shift(K=8, n_per_class=200, shift=0.6, seed=0)
    reference = expected_min_coverage(8, 50, 50, alpha=0.10, reps=4000).mean
    kwargs = dict(k_values=[8], subset_seeds=20, n_cal_per_class=50, alpha=0.10,
                  methods=["mondrian", "oracle"], design="nested")

    def mean_worst(records, method):
        return float(np.mean([r.worst_class for r in records if r.method == method]))

    view = cks.no_shift_view(ds, view_seed=0)
    no_shift = cks.run_sweep(view, **kwargs)
    assert all(r.n_eval_total == 50 * 8 for r in no_shift)
    assert abs(mean_worst(no_shift, "mondrian") - reference) < 0.06
    assert abs(mean_worst(no_shift, "oracle") - reference) < 0.06

    shifted = cks.run_sweep(ds, **kwargs)
    assert mean_worst(shifted, "mondrian") < reference - 0.3


_BASE_ARGS = ["--cache", "synthetic", "--k-values", "4,8", "--subset-seeds", "2",
              "--n-cal-per-class", "20", "--methods", "split,mondrian,oracle"]
_RELEASED_CONFIG_KEYS = ["cache", "k_values", "subset_seeds", "n_cal_per_class", "alpha",
                         "eval_frac"]


def test_default_command_line_writes_the_released_schema(tmp_path, capsys):
    out = tmp_path / "default.json"
    assert cks.main(_BASE_ARGS + ["--out", str(out)]) == 0
    capsys.readouterr()
    payload = json.loads(out.read_text())
    assert list(payload) == ["config", "records", "summary"]
    assert list(payload["config"]) == _RELEASED_CONFIG_KEYS
    ds = make_synthetic_shift()
    records = cks.run_sweep(ds, [4, 8], 2, 20, 0.10, ["split", "mondrian", "oracle"], 0.5)
    released = cks.build_json_payload("synthetic", [4, 8], 2, 20, 0.10, records,
                                      cks.summarize(records), 0.5)
    assert out.read_text() == json.dumps(released, indent=2) + "\n"
    assert [p.name for p in tmp_path.iterdir()] == ["default.json"]


def test_nested_command_line_records_the_design_the_orders_and_the_provenance(tmp_path, capsys):
    out = tmp_path / "nested.json.gz"
    args = _BASE_ARGS + ["--subset-design", "nested", "--stamp", "abc1234", "--out", str(out)]
    assert cks.main(args) == 0
    capsys.readouterr()
    with gzip.open(out, "rt") as handle:
        payload = json.load(handle)
    assert list(payload) == ["config", "records", "summary", "class_orders", "problems",
                             "provenance"]
    config = payload["config"]
    assert list(config)[:len(_RELEASED_CONFIG_KEYS)] == _RELEASED_CONFIG_KEYS
    assert config["subset_design"] == "nested"
    assert config["view"] == "shift" and config["view_seed"] is None
    assert config["method_seed"] == 0
    assert len(payload["class_orders"]) == 2
    for seed, order in enumerate(payload["class_orders"]):
        assert order == cks.nested_class_order(12, seed).tolist()
    provenance = payload["provenance"]
    assert provenance["code_version"] == "abc1234"
    assert provenance["numpy"] == np.__version__
    assert set(provenance) == {"code_version", "python", "numpy", "sklearn",
                               "source_cal_sha256", "n_source_cal", "n_target_test",
                               "n_classes"}
    assert len(provenance["source_cal_sha256"]) == 64
    assert provenance["n_classes"] == 12 and provenance["n_source_cal"] == 12 * 500
    assert len(payload["records"]) == 2 * 2 * 3
    problems = {(q["k"], q["subset_seed"]): q for q in payload["problems"]}
    assert sorted(problems) == [(4, 0), (4, 1), (8, 0), (8, 1)]
    for (k, seed), q in problems.items():
        assert q["class_ids"] == sorted(payload["class_orders"][seed][:k])
        assert q["n_eval_per_class"] == [250] * k
        assert len(q["eval_top1_per_class"]) == k
        assert all(0.0 <= a <= 1.0 for a in q["eval_top1_per_class"])
    for r in payload["records"]:
        assert len(r["per_class_coverage"]) == len(r["n_cal_per_class"]) == r["k"]
        own = 250 if r["method"] == "oracle" else 20
        assert r["n_cal_per_class"] == [own] * r["k"]
    first = out.read_bytes()
    assert cks.main(args) == 0
    capsys.readouterr()
    assert out.read_bytes() == first
    assert [p.name for p in tmp_path.iterdir()] == ["nested.json.gz"]


def test_no_shift_command_line_records_the_view(tmp_path, capsys):
    out = tmp_path / "noshift.json.gz"
    args = _BASE_ARGS + ["--subset-design", "nested", "--view", "no_shift", "--view-seed",
                         "3", "--out", str(out)]
    assert cks.main(args) == 0
    capsys.readouterr()
    with gzip.open(out, "rt") as handle:
        payload = json.load(handle)
    assert payload["config"]["view"] == "no_shift"
    assert payload["config"]["view_seed"] == 3
    assert payload["provenance"]["code_version"] is None
    assert payload["provenance"]["n_source_cal"] == 12 * 500
    for r in payload["records"]:
        assert r["n_eval_total"] == 50 * r["k"]


def test_the_source_hash_names_the_source_rows_and_nothing_else():
    ds = make_synthetic_shift(K=8, n_per_class=60, shift=0.6, seed=0)
    same_source = make_synthetic_shift(K=8, n_per_class=60, shift=0.6, seed=0)
    target_moved = dataclasses.replace(
        same_source,
        logits=np.where((same_source.split == TARGET_TEST)[:, None], same_source.logits + 1.0,
                        same_source.logits).astype(np.float32))
    other_source = make_synthetic_shift(K=8, n_per_class=60, shift=0.6, seed=1)
    digest = cks.source_cal_digest(ds)
    assert digest == cks.source_cal_digest(target_moved)
    assert digest != cks.source_cal_digest(other_source)


def test_a_stamp_alone_keeps_the_released_schema(tmp_path, capsys):
    """Only the nested design or the no-shift view extends the file."""
    plain, stamped = tmp_path / "plain.json", tmp_path / "stamped.json"
    assert cks.main(_BASE_ARGS + ["--out", str(plain)]) == 0
    assert cks.main(_BASE_ARGS + ["--stamp", "abc1234", "--out", str(stamped)]) == 0
    capsys.readouterr()
    assert stamped.read_bytes() == plain.read_bytes()


def test_the_problem_facts_describe_the_rows_the_methods_were_scored_on():
    """The top-1 accuracy stored per class is that of the evaluation rows, with the classes in
    the order of the per-class coverage vector."""
    ds = make_synthetic_shift(K=12, n_per_class=120, shift=0.8, seed=0)
    extras = cks.new_extras()
    records = cks.run_sweep(ds, [6], 1, 20, 0.10, ["oracle"], 0.5, "nested", extras)
    (problem,) = extras["problems"]
    sub, _ = cks.build_k_subproblem(ds, 6, 0, 20, "nested")
    logits, labels = sub.target_test()
    _, eval_idx = cks.split_target_labeled_and_eval(
        logits, labels, 6, 0, 0.5, class_ids=np.asarray(sub.meta["class_subset"]))
    hit = logits[eval_idx].argmax(axis=1) == labels[eval_idx]
    want = [float(hit[labels[eval_idx] == c].mean()) for c in range(6)]
    assert problem["eval_top1_per_class"] == want
    assert problem["class_ids"] == sub.meta["class_subset"]
    assert extras["n_cal_per_class"][(6, 0, "oracle")] == [60] * 6
    assert len(records) == 1


def test_two_writers_of_one_output_use_different_temporary_names(tmp_path, monkeypatch):
    seen = []
    real_replace = cks.os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(cks.os, "replace", spy)
    monkeypatch.setattr(cks.os, "getpid", lambda: 111)
    cks.write_payload({"config": {}}, tmp_path / "x.json.gz")
    monkeypatch.setattr(cks.os, "getpid", lambda: 222)
    cks.write_payload({"config": {}}, tmp_path / "x.json.gz")
    assert seen == ["x.json.gz.tmp111", "x.json.gz.tmp222"]


def test_a_killed_write_leaves_no_file_under_the_output_name(tmp_path, monkeypatch):
    out = tmp_path / "killed.json.gz"

    def boom(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(cks.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        cks.write_payload({"config": {}}, out)
    assert not out.exists()
