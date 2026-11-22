"""Tests for scripts/sweep_uncertainty.py (a standalone script, not part of the installed
``pcpbench`` package -- loaded here by file path, matching the convention in
test_controlled_k_sweep.py and test_label_complexity_sweep.py)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "sweep_uncertainty.py"
_ANALYSIS_DIR = _ROOT / "outputs" / "analysis"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_uncertainty", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


su = _load_module()


def test_constant_matrix_gives_point_mean_and_zero_width_interval() -> None:
    mat = np.full((5, 5), 0.42)
    rng = np.random.default_rng(0)
    draws = su.bootstrap_two_level_mean(mat, rng, 2000)
    s = su.summarize(float(mat.mean()), draws)
    assert s["mean"] == pytest.approx(0.42)
    assert s["ci_lo"] == pytest.approx(0.42)
    assert s["ci_hi"] == pytest.approx(0.42)
    assert s["se"] == 0.0


def test_constant_shift_between_k10_and_k100_gives_exact_decay() -> None:
    d = 0.137
    caches = ("cA", "cB")
    idx: dict = {}
    for method, base, other_base in (("foo", 0.9, 0.5), ("oracle", 0.8, 0.5)):
        shift = d if method == "foo" else 0.021
        for k, val in ((10, base), (100, base - shift)):
            for cache in caches:
                for seed in range(5):
                    idx.setdefault(method, {}).setdefault(k, {}).setdefault(cache, {})[seed] = val
    rng = np.random.default_rng(0)
    decay, caveat = su.controlled_decay(idx, caches, ("foo", "oracle"), 5, rng, 500)
    entry = next(x for x in decay if x.method == "foo")
    assert entry.decay_mean == pytest.approx(d, abs=1e-9)
    assert entry.ratio_to_oracle == pytest.approx(d / 0.021, abs=1e-9)
    assert entry.paper_convention_ratio == pytest.approx(d / 0.021, abs=1e-9)
    assert caveat.B == 500
    assert caveat.methods == ("foo",)
    assert caveat.n_oracle_decay_draws_le_0_01 == (0,)
    assert caveat.n_oracle_decay_draws_le_0 == (0,)


def test_two_level_bootstrap_is_wider_than_one_level_pooled_bootstrap() -> None:
    mat = np.array([[0.0] * 5, [1.0] * 5])
    rng_two = np.random.default_rng(0)
    two_level_draws = su.bootstrap_two_level_mean(mat, rng_two, 8000)
    rng_one = np.random.default_rng(0)
    one_level_draws = su.bootstrap_one_level_mean(mat.ravel(), rng_one, 8000)
    width_two = np.percentile(two_level_draws, 97.5) - np.percentile(two_level_draws, 2.5)
    width_one = np.percentile(one_level_draws, 97.5) - np.percentile(one_level_draws, 2.5)
    assert width_two > width_one


def test_reproduces_label_sweep_paper_values() -> None:
    path = _ANALYSIS_DIR / f"label_complexity_{su.LABEL_SWEEP_CACHE}.json"
    if not path.exists():
        pytest.skip("committed label_complexity_contrast_s3.json cache is absent from outputs/analysis")

    payload = su.load_label_sweep(_ANALYSIS_DIR, su.LABEL_SWEEP_CACHE)
    rng = np.random.default_rng(0)
    stats, _pairs, _smallest_ppi, _smallest_oracle = su.label_sweep_stats(
        payload, su.LABEL_SWEEP_METHODS, rng, 200
    )
    expected_ppi = {10: 0.460, 25: 0.504, 50: 0.696}
    expected_oracle = {10: 0.592, 25: 0.708, 50: 0.724}
    for m, v in expected_ppi.items():
        entry = next(s for s in stats if s.method == "ppi" and s.m == m)
        assert round(entry.mean, 3) == v
    for m, v in expected_oracle.items():
        entry = next(s for s in stats if s.method == "oracle" and s.m == m)
        assert round(entry.mean, 3) == v


def test_label_sweep_floor_at_capped_m_equals_floor_at_m50() -> None:
    path = _ANALYSIS_DIR / f"label_complexity_{su.LABEL_SWEEP_CACHE}.json"
    if not path.exists():
        pytest.skip("committed label_complexity_contrast_s3.json cache is absent from outputs/analysis")

    payload = su.load_label_sweep(_ANALYSIS_DIR, su.LABEL_SWEEP_CACHE)
    rng = np.random.default_rng(0)
    stats, _pairs, _smallest_ppi, _smallest_oracle = su.label_sweep_stats(
        payload, su.LABEL_SWEEP_METHODS, rng, 200
    )
    by_method_m = {(s.method, s.m): s for s in stats}
    for method in su.LABEL_SWEEP_METHODS:
        floor_50 = by_method_m[(method, 50)].floor_mean
        for m in (100, 200):
            entry = by_method_m[(method, m)]
            assert entry.floor_mean == pytest.approx(floor_50)
            assert entry.excess == pytest.approx(entry.mean - floor_50)


def test_label_sweep_labels_per_class_drawn_matches_artifact() -> None:
    path = _ANALYSIS_DIR / f"label_complexity_{su.LABEL_SWEEP_CACHE}.json"
    if not path.exists():
        pytest.skip("committed label_complexity_contrast_s3.json cache is absent from outputs/analysis")

    payload = su.load_label_sweep(_ANALYSIS_DIR, su.LABEL_SWEEP_CACHE)

    counts_by_m: dict[int, set[int]] = {}
    for r in payload["per_seed"]:
        counts_by_m.setdefault(r["m"], set()).update(r["per_class_labeled"])

    rng = np.random.default_rng(0)
    stats, _pairs, _smallest_ppi, _smallest_oracle = su.label_sweep_stats(
        payload, su.LABEL_SWEEP_METHODS, rng, 200
    )
    for m, counts in counts_by_m.items():
        if m < 1:
            continue
        assert len(counts) == 1, "fixture assumption: a single constant count per m"
        expected_count = next(iter(counts))
        for method in su.LABEL_SWEEP_METHODS:
            entry = next(s for s in stats if s.method == method and s.m == m)
            assert entry.labels_per_class_drawn == expected_count


def test_label_sweep_floor_falls_back_when_counts_disagree() -> None:
    K = su.K_LABEL_SWEEP
    methods = ("ppi", "oracle", "acp")
    per_seed = []
    disagreeing_arrays = {0: [3] * K, 1: [5] * K}
    for method in methods:
        for seed, arr in disagreeing_arrays.items():
            per_seed.append({
                "method": method, "m": 1, "seed": seed,
                "per_class_labeled": arr, "worst_class": 0.5,
            })
    payload = {"m_values": [1], "per_seed": per_seed}

    rng = np.random.default_rng(0)
    stats, _pairs, _smallest_ppi, _smallest_oracle = su.label_sweep_stats(payload, methods, rng, 50)
    entry = next(s for s in stats if s.method == "ppi" and s.m == 1)
    assert entry.cal_source == "fallback_per_seed_average"

    from pcpbench.metrics.validity_floor import expected_min_coverage
    f0 = expected_min_coverage(K, np.array([3] * K), su.N_EVAL_LABEL_SWEEP, alpha=su.ALPHA)
    f1 = expected_min_coverage(K, np.array([5] * K), su.N_EVAL_LABEL_SWEEP, alpha=su.ALPHA)
    expected_floor_mean = (f0.mean + f1.mean) / 2.0
    assert entry.floor_mean == pytest.approx(expected_floor_mean)


def test_count_draws_at_or_below() -> None:
    draws = np.array([-0.5, -0.1, 0.0, 0.005, 0.02, 1.0])
    assert su.count_draws_at_or_below(draws, 0.01) == 4
    assert su.count_draws_at_or_below(draws, 0.0) == 3


def _join_and_expected(items: list[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def test_markdown_drops_ratio_se_states_cap_and_caveat() -> None:
    cache_paths = [_ANALYSIS_DIR / f"controlled_k_sweep_{c}.json" for c in su.MODERATE_CACHES]
    cache_paths += [_ANALYSIS_DIR / f"controlled_k_sweep_{c}.json" for c in su.FLOORED_CACHES]
    label_path = _ANALYSIS_DIR / f"label_complexity_{su.LABEL_SWEEP_CACHE}.json"
    if not all(p.exists() for p in cache_paths) or not label_path.exists():
        pytest.skip("committed sweep caches are absent from outputs/analysis")

    payload, md = su.run_all(_ANALYSIS_DIR, B=200, seed=0)

    assert "ratio SE" not in md
    assert "100 target labels per class" not in md
    assert "200 target labels per class" not in md
    assert "labels per class drawn" in md
    assert "| displayed-endpoint convention |" in md
    assert "paper convention" not in md

    caveat = payload["controlled"]["ratio_se_caveat"]
    assert tuple(caveat["methods"]) == ("split", "mondrian", "weighted", "ppi", "acp")
    expected_sentence = (
        "In the draws behind the five ratio intervals the oracle's decay fell at or "
        f"below 0.01 in {_join_and_expected([str(c) for c in caveat['n_oracle_decay_draws_le_0_01']])} "
        f"of {caveat['B']} draws for {_join_and_expected(caveat['methods'])}, and at or "
        f"below zero in {_join_and_expected([str(c) for c in caveat['n_oracle_decay_draws_le_0']])}."
    )
    assert expected_sentence in md

    ppi_m0 = next(s for s in payload["label_sweep"]["per_m"] if s["method"] == "ppi" and s["m"] == 0)
    assert ppi_m0["labels_per_class_drawn"] == 0
    assert ppi_m0["floor_mean"] is None
    assert "| ppi | 0 | 0 |" in md

    rows = {(s["method"], s["m"]): s for s in payload["label_sweep"]["per_m"]}
    capped = [s for s in payload["label_sweep"]["per_m"] if s.get("reused_from_m") is not None]
    assert capped, "fixture assumption: at least one row is capped in the real artifact"
    for s in capped:
        source = rows[(s["method"], s["reused_from_m"])]
        assert s["mean"] == source["mean"]
        assert s["se"] == source["se"]
        assert s["ci_lo"] == source["ci_lo"]
        assert s["ci_hi"] == source["ci_hi"]
        assert s["floor_mean"] == source["floor_mean"]
        assert s["floor_sd"] == source["floor_sd"]
        assert s["excess"] == source["excess"]
        assert (
            f"repeat the m equal to {s['reused_from_m']} row exactly, statistics included"
            in md
        )

    pair_rows = {p["m"]: p for p in payload["label_sweep"]["ppi_vs_oracle"]}
    capped_pairs = [p for p in payload["label_sweep"]["ppi_vs_oracle"] if p.get("reused_from_m") is not None]
    assert capped_pairs, "fixture assumption: at least one paired-difference row is capped"
    for p in capped_pairs:
        source = pair_rows[p["reused_from_m"]]
        assert p["diff_mean"] == source["diff_mean"]
        assert p["diff_se"] == source["diff_se"]
        assert p["diff_ci_lo"] == source["diff_ci_lo"]
        assert p["diff_ci_hi"] == source["diff_ci_hi"]

    assert "up to independent bootstrap resampling noise" not in md


def test_range_end_sentence_stops_on_a_missing_or_reversed_end() -> None:
    payload, _md = su.run_all(_ANALYSIS_DIR, B=200, seed=0)
    decay = [su.DecayStat(**d) for d in payload["controlled"]["decay"]]
    with pytest.raises(ValueError, match="no decay ratio for ppi"):
        su._range_end_sentences([d for d in decay if d.method != "ppi"])
    swapped = [dataclasses.replace(d, method={"mondrian": "ppi", "ppi": "mondrian"}.get(d.method, d.method))
               for d in decay]
    with pytest.raises(ValueError, match="not the lower and the upper end"):
        su._range_end_sentences(swapped)
