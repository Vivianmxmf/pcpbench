"""Tests for scripts/class_count_trend.py, the comparison of every K-sweep file with the matched
finite-sample reference, for the released independent design and for the nested design."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

from pcpbench.runner import METHODS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _sweep_payloads  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "class_count_trend.py"
_ANALYSIS = _ROOT / "outputs" / "analysis"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cct = _load_script(_SCRIPT_PATH, "class_count_trend")

ALL_METHODS = tuple(sorted(METHODS))
REPS = 2000


@pytest.fixture(scope="module")
def reference():
    return cct.reference_table(cct.K_VALUES, reps=REPS)


_order = _sweep_payloads.class_order
_payload = _sweep_payloads.make_payload


def test_the_shared_payload_builder_follows_the_script_constants():
    assert _sweep_payloads.K_VALUES == cct.K_VALUES
    assert _sweep_payloads.N_CLASSES == cct.N_CLASSES
    assert _sweep_payloads.ALL_METHODS == ALL_METHODS


def test_the_required_acp_tie_rule_is_the_calibrator_default():
    """A nested file must say that ACP broke its ties at random; the rule is the calibrator's
    default, so a change of either must be made in both places."""
    from pcpbench.calibrators.acp import DEFAULT_TIE_BREAK

    assert cct.ACP_TIE_BREAK == DEFAULT_TIE_BREAK == "random"


def _grid(*, design="nested", n_seeds=3, coverage=None, backbones=cct.BACKBONES):
    return [
        cct.as_sweep(f"{backbone}/{corruption}_s{severity}",
                     _payload(backbone, corruption, severity, design=design, n_seeds=n_seeds,
                              coverage=coverage))
        for backbone in backbones
        for corruption in cct.CORRUPTIONS
        for severity in cct.SEVERITIES
    ]


def _row(rows, **match):
    found = [r for r in rows if all(r[key] == value for key, value in match.items())]
    assert len(found) == 1, (match, len(found))
    return found[0]


def test_cache_key_reads_backbone_corruption_and_severity():
    assert cct.cache_key("caches/cifar100c_vitL_jpeg_compression_s4.npz") == (
        "ViT-L/14", "jpeg_compression", 4)
    assert cct.cache_key("caches/cifar100c_fog_s1.npz") == ("ViT-B/32", "fog", 1)
    assert cct.cache_key("/abs/elsewhere/cifar100c_gaussian_noise_s5.npz") == (
        "ViT-B/32", "gaussian_noise", 5)
    for name in ("caches/cifar100c.npz", "caches/cifar100c_mild.npz", "caches/cifar100c_vitL.npz",
                 "caches/cifar10c_fog_s1.npz", "caches/cifar100c_snow_s1.npz", "synthetic"):
        with pytest.raises(ValueError):
            cct.cache_key(name)


def test_load_payload_reads_plain_and_gzipped_json(tmp_path):
    payload = {"config": {"cache": "x"}, "records": [1, 2]}
    plain = tmp_path / "a.json"
    plain.write_text(json.dumps(payload))
    zipped = tmp_path / "a.json.gz"
    with gzip.open(zipped, "wt") as handle:
        json.dump(payload, handle)
    assert cct.load_payload(plain) == payload
    assert cct.load_payload(zipped) == payload


def test_reference_declines_with_k_and_matches_the_committed_floor():
    table = cct.reference_table(cct.K_VALUES)
    means = [table[k].mean for k in cct.K_VALUES]
    assert means == sorted(means, reverse=True)
    committed = _ANALYSIS / "validity_floor.json"
    if not committed.exists():
        pytest.skip("validity_floor.json is not in this checkout")
    rows = {r["K"]: r for r in json.loads(committed.read_text())["rows"]
            if r["setting"] == "k_sweep"}
    for k in cct.K_VALUES:
        assert table[k].mean == pytest.approx(rows[k]["floor_mean"], abs=1e-12)
        assert table[k].sd == pytest.approx(rows[k]["floor_sd"], abs=1e-12)


def test_seed_interval_is_the_t_interval_on_the_seed_means():
    values = np.array([0.1, 0.2, 0.4, 0.5, 0.3])
    lo, hi = cct.seed_interval(values, 0.95)
    half = scipy_stats.t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5)
    assert (lo, hi) == pytest.approx((0.3 - half, 0.3 + half))
    wide_lo, wide_hi = cct.seed_interval(values, cct.BONFERRONI_LEVEL)
    assert wide_lo < lo and wide_hi > hi
    assert cct.seed_interval(np.full(6, 0.25), 0.95) == pytest.approx((0.25, 0.25))
    with pytest.raises(ValueError):
        cct.seed_interval(np.array([0.5]), 0.95)


def test_seed_interval_holds_its_level_when_caches_differ_far_more_than_seeds():
    """With caches as fixed conditions and seeds as replicates, the interval covers the mean over
    those caches at its nominal rate however much the caches differ."""
    rng = np.random.default_rng(0)
    cache_effect = rng.normal(0.0, 5.0, size=40)
    target = cache_effect.mean()
    trials, misses, wide_misses = 4000, 0, 0
    for _ in range(trials):
        cells = cache_effect[:, None] + rng.normal(0.0, 1.0, size=(40, 30))
        lo, hi = cct.seed_interval(cells.mean(axis=0), 0.95)
        misses += not lo <= target <= hi
        lo, hi = cct.seed_interval(cells.mean(axis=0), cct.BONFERRONI_LEVEL)
        wide_misses += not lo <= target <= hi
    assert 0.035 < misses / trials < 0.065
    assert wide_misses / trials < 3 * (1 - cct.BONFERRONI_LEVEL) + 0.002


def test_the_family_is_fixed_and_leaves_clustered_and_the_oracle_out():
    assert cct.FAMILY_SIZE == 20
    assert cct.BONFERRONI_LEVEL == pytest.approx(1 - 0.05 / 20)
    assert set(cct.FAMILY_METHODS) == set(ALL_METHODS) - {"oracle", "clustered"}
    names = [scope.name for scope in cct.analysis_scopes()]
    assert names == ["ViT-B/32", "ViT-L/14", cct.MODERATE_SCOPE]
    assert [scope.by_family for scope in cct.analysis_scopes()] == [True, True, False]


def test_band_counts_split_the_cache_means():
    counts = cct.band_counts(np.array([0.1, 0.5, 0.69, 0.71, 0.9]), 0.7, 0.05)
    assert counts == {"below": 2, "within": 2, "above": 1, "half_width": 0.05}


def test_expected_subset_min_is_the_mean_over_every_subset():
    """The Rao-Blackwellized minimum: the exact mean, over every subset of m of the values, of
    the subset's minimum."""
    import itertools

    values = np.array([0.7, 0.2, 0.9, 0.2, 0.5, 1.0, 0.0])
    for m in range(1, values.size + 1):
        brute = np.mean([min(c) for c in itertools.combinations(values, m)])
        assert cct.expected_subset_min(values, m) == pytest.approx(brute, abs=1e-12), m
    assert cct.expected_subset_min(values, 1) == pytest.approx(values.mean())
    assert cct.expected_subset_min(values, values.size) == pytest.approx(values.min())
    v = np.r_[np.full(10, 0.5), np.full(90, 0.9)]
    miss = math.comb(90, 10) / math.comb(100, 10)
    assert cct.expected_subset_min(v, 10) == pytest.approx(0.5 + 0.4 * miss, abs=1e-12)
    for m in (0, 101):
        with pytest.raises(ValueError):
            cct.expected_subset_min(v, m)


def test_skewness_is_the_adjusted_sample_skewness_and_absent_without_spread():
    values = np.array([0.0, 0.0, 0.0, -0.3, 0.1])
    assert cct.skewness(values) == pytest.approx(scipy_stats.skew(values, bias=False))
    assert cct.skewness(np.full(5, 0.083)) is None
    assert cct.skewness(np.array([0.1, 0.2])) is None


def test_per_k_rows_report_mean_difference_zero_share_and_set_size(reference):
    def coverage(method, k, seed, ids):
        if method == "split":
            return np.zeros(k)
        if method == "oracle":
            return np.full(k, reference[k].mean)
        if method == "mondrian":
            return np.full(k, reference[k].mean - 0.3)
        return np.full(k, 0.95)

    result = cct.analyse(_grid(coverage=coverage), design="nested", reps=REPS)
    assert result["config"]["n_seeds"] == 3
    assert list(result["scopes"]) == ["ViT-B/32", "ViT-L/14", cct.MODERATE_SCOPE]
    for name, n_caches in (("ViT-B/32", 40), ("ViT-L/14", 40), (cct.MODERATE_SCOPE, 5)):
        scope = result["scopes"][name]
        assert scope["n_caches"] == n_caches == len(scope["caches"])
        assert scope["selected_on_outcomes"] is (name == cct.MODERATE_SCOPE)
        for k in cct.K_VALUES:
            split = _row(scope["per_k"], method="split", k=k)
            assert split["mean"] == 0.0 and split["zero_share"] == 1.0
            assert split["diff"] == pytest.approx(-reference[k].mean)
            oracle = _row(scope["per_k"], method="oracle", k=k)
            assert oracle["diff"] == pytest.approx(0.0, abs=1e-12)
            assert oracle["zero_share"] == 0.0
            mondrian = _row(scope["per_k"], method="mondrian", k=k)
            assert mondrian["diff"] == pytest.approx(-0.3)
            assert mondrian["ci_lo"] == pytest.approx(mondrian["mean"])
            assert mondrian["ci_hi"] == pytest.approx(mondrian["mean"])
            assert _row(scope["per_k"], method="acp", k=k)["set_over_k"] == pytest.approx(0.2)
            assert "band_seed_mean" not in split and "band_single_draw" not in split
    assert "family_min" in result["scopes"]["ViT-B/32"]["per_k"][0]
    assert "family_min" not in result["scopes"][cct.MODERATE_SCOPE]["per_k"][0]


def test_an_interval_reflects_the_seeds_and_not_the_spread_between_caches():
    """Caches differ by far more than seeds do."""
    payloads = []
    for backbone in cct.BACKBONES:
        for ci, corruption in enumerate(cct.CORRUPTIONS):
            for severity in cct.SEVERITIES:
                def coverage(method, k, seed, ids, ci=ci):
                    return np.full(k, 0.1 * ci + 0.01 * seed)
                payloads.append(cct.as_sweep("x", _payload(backbone, corruption, severity,
                                                           n_seeds=4, coverage=coverage)))
    result = cct.analyse(payloads, design="nested", reps=REPS)
    row = _row(result["scopes"]["ViT-B/32"]["per_k"], method="mondrian", k=25)
    assert row["mean"] == pytest.approx(0.35 + 0.015)
    seed_means = 0.35 + 0.01 * np.arange(4)
    assert (row["ci_lo"], row["ci_hi"]) == pytest.approx(cct.seed_interval(seed_means, 0.95))
    assert row["ci_hi"] - row["ci_lo"] < 0.05
    assert row["family_min"] == pytest.approx(0.015)
    assert row["family_max"] == pytest.approx(0.715)


def test_a_coverage_interval_stays_inside_zero_and_one(reference):
    def coverage(method, k, seed, ids):
        return np.full(k, 0.3 if seed == 0 else 0.0)

    result = cct.analyse(_grid(coverage=coverage), design="nested", reps=REPS)
    row = _row(result["scopes"]["ViT-B/32"]["per_k"], method="ppi", k=10)
    assert row["mean"] == pytest.approx(0.1)
    assert row["ci_lo"] == 0.0 and row["ci_hi"] <= 1.0


def _planted(method, k, seed, ids):
    if method != "mondrian":
        return np.full(k, 0.9)
    if k == 10:
        return np.full(k, 0.8)
    first_ten = set(_order(seed)[:10])
    return np.array([0.7 if g in first_ten else 0.5 for g in ids])


def test_decomposition_recovers_planted_terms_and_they_sum_to_the_total(reference):
    result = cct.analyse(_grid(coverage=_planted), design="nested", reps=REPS)
    step = reference[100].mean - reference[10].mean
    for name in cct.BACKBONES:
        rows = result["scopes"][name]["decomposition"]
        assert {(r["k_small"], r["k_large"]) for r in rows} == {(10, 100)}
        row = _row(rows, method="mondrian")
        assert row["same_class"]["mean"] == pytest.approx(-0.1)
        assert row["selection"]["mean"] == pytest.approx(-0.2 - step)
        assert row["total"] == pytest.approx(-0.3 - step)
        assert row["total"] == pytest.approx(row["same_class"]["mean"] + row["selection"]["mean"])
        assert row["mean_class_change"] == pytest.approx(-0.1)
        assert row["same_class"]["n_caches_negative"] == 40 == row["same_class"]["n_caches"]
        assert row["same_class"]["n_families_negative"] == 8
        assert row["same_class"]["family_min"] == pytest.approx(-0.1)
        assert row["in_family"] is True
        assert row["same_class"]["bonferroni_lo"] <= row["same_class"]["ci_lo"]
        assert row["same_class"]["bonferroni_hi"] >= row["same_class"]["ci_hi"]
        for term in cct.FAMILY_TERMS:
            assert row[term]["marked"] is False and row[term]["bound_share"] == 0.0
            assert row[term]["n_free"] == 40 * 3
            assert row[term]["free_mean"] == pytest.approx(row[term]["mean"])
        assert row["selection"]["pinned_part"] == 0.0
        assert row["limits"] == {name: {"at_zero": 0.0, "at_one": 0.0}
                                 for name in ("worst_small", "kept_min_large", "worst_large")}
        assert row["same_class"]["excludes_zero"] == -1 == row["selection"]["excludes_zero"]
        flat = _row(rows, method="ppi")
        assert flat["same_class"]["mean"] == pytest.approx(0.0)
        assert flat["selection"]["mean"] == pytest.approx(-step)
        assert flat["same_class"]["n_caches_negative"] == 0
    family = result["config"]["family"]
    assert family["size"] == 20 and family["level"] == pytest.approx(1 - 0.05 / 20)
    assert family["k_pair"] == [10, 100]


def test_only_the_five_headline_methods_on_the_backbone_scopes_are_in_the_family():
    result = cct.analyse(_grid(coverage=_planted), design="nested", reps=REPS)
    in_family = [(name, row["method"]) for name, scope in result["scopes"].items()
                 for row in scope["decomposition"] if row["in_family"]]
    assert len(in_family) * len(cct.FAMILY_TERMS) == cct.FAMILY_SIZE
    assert {name for name, _ in in_family} == set(cct.BACKBONES)
    for name, scope in result["scopes"].items():
        for row in scope["decomposition"]:
            outside = name == cct.MODERATE_SCOPE or row["method"] in ("oracle", "clustered")
            assert row["in_family"] is (not outside)
            for term in cct.FAMILY_TERMS:
                assert (row[term]["bonferroni_lo"] is None) is outside
    moderate = _row(result["scopes"][cct.MODERATE_SCOPE]["decomposition"], method="mondrian")
    assert moderate["same_class"]["mean"] == pytest.approx(-0.1)
    assert moderate["same_class"]["n_caches"] == 5
    assert "n_families_negative" not in moderate["same_class"]


def test_a_floored_method_is_marked_and_its_artefact_is_explained(reference):
    def coverage(method, k, seed, ids):
        return np.zeros(k) if method == "split" else np.full(k, 0.9)

    result = cct.analyse(_grid(coverage=coverage), design="nested", reps=REPS)
    row = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="split")
    step = reference[100].mean - reference[10].mean
    assert row["same_class"]["mean"] == 0.0
    assert row["total"] == pytest.approx(-step)
    assert row["total"] > 0
    assert row["zero_share_k_small"] == 1.0 and row["zero_share_k_large"] == 1.0
    for term in cct.FAMILY_TERMS:
        assert row[term]["marked"] is True and row[term]["bound_share"] == 1.0
        assert row[term]["n_free"] == 0 and row[term]["free_mean"] is None
    assert row["selection"]["pinned_part"] == pytest.approx(row["selection"]["mean"])
    assert row["limits"]["kept_min_large"] == {"at_zero": 1.0, "at_one": 0.0}
    flat = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="ppi")
    assert flat["same_class"]["marked"] is False and flat["selection"]["marked"] is False
    markdown = cct.render_markdown(result)
    assert "same-class term bound, selection term bound" in markdown
    assert "outside the family" in markdown
    assert "floor-bound" not in markdown


def _ppi_like(method, k, seed, ids):
    if method != "ppi":
        return np.full(k, 0.9)
    order = _order(seed)
    if k == 10:
        values = np.full(k, 0.6)
        if seed == 0:
            values[0] = 0.0
        return values
    zero = set(order[66:]) if seed == 3 else {order[0], *order[67:]}
    return np.array([0.0 if g in zero else 0.4 for g in ids])


def test_the_selection_term_is_marked_where_the_kept_minimum_is_pinned(reference):
    """A rule on the K 10 zero share, which is 0.25 here, would mark nothing, while three cells
    in four have a selection term fixed at minus the reference's decline whatever the method
    does."""
    result = cct.analyse(_grid(coverage=_ppi_like, n_seeds=4), design="nested", reps=REPS)
    step = reference[100].mean - reference[10].mean
    row = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="ppi")
    assert row["zero_share_k_small"] == 0.25
    assert row["limits"] == {"worst_small": {"at_zero": 0.25, "at_one": 0.0},
                             "kept_min_large": {"at_zero": 0.75, "at_one": 0.0},
                             "worst_large": {"at_zero": 1.0, "at_one": 0.0}}
    selection = row["selection"]
    assert selection["bound_share"] == 0.75 and selection["marked"] is True
    assert selection["pinned_part"] == pytest.approx(0.75 * -step)
    assert selection["n_free"] == 40 and selection["free_mean"] == pytest.approx(-0.4 - step)
    assert selection["mean"] == pytest.approx(selection["pinned_part"] + 0.25 * (-0.4 - step))
    same = row["same_class"]
    assert same["bound_share"] == 0.25 and same["marked"] is False
    assert same["n_free"] == 120 and same["free_mean"] == pytest.approx((-0.6 - 0.6 - 0.2) / 3)
    markdown = cct.render_markdown(result)
    line = next(l for l in markdown.splitlines() if l.startswith("| PPI") and "{" in l)
    assert line.rstrip(" |").endswith("selection term bound")


def test_a_method_at_the_ceiling_is_marked_as_well(reference):
    """ACP as the released files show it: full coverage of its worst class at K 10 in most cells,
    which makes its same-class term one-sided, and full coverage of every class at K 100 in some,
    which pins its selection term."""
    def coverage(method, k, seed, ids):
        if method != "acp":
            return np.full(k, 0.9)
        if k == 10:
            return np.full(k, 0.9 if seed == 3 else 1.0)
        kept = set(_order(seed)[:10])
        return np.array([1.0 if seed == 0 or g in kept else 0.8 for g in ids])

    result = cct.analyse(_grid(coverage=coverage, n_seeds=4), design="nested", reps=REPS)
    step = reference[100].mean - reference[10].mean
    row = _row(result["scopes"]["ViT-L/14"]["decomposition"], method="acp")
    assert row["limits"] == {"worst_small": {"at_zero": 0.0, "at_one": 0.75},
                             "kept_min_large": {"at_zero": 0.0, "at_one": 1.0},
                             "worst_large": {"at_zero": 0.0, "at_one": 0.25}}
    assert row["same_class"]["bound_share"] == 0.75 and row["same_class"]["marked"] is True
    assert row["same_class"]["free_mean"] == pytest.approx(0.1)
    assert row["selection"]["bound_share"] == 0.25 and row["selection"]["marked"] is False
    assert row["selection"]["free_mean"] == pytest.approx(-0.2 - step)
    assert row["selection"]["pinned_part"] == pytest.approx(0.25 * -step)


def test_the_expectation_version_flags_an_exclusion_that_rests_on_the_drawn_subset(reference):
    """Source-Mondrian's planted selection term is negative only because the kept ten classes
    read higher than the rest, which a random ten would not."""
    def coverage(method, k, seed, ids):
        if method == "weighted" and k == 100:
            last = _order(seed)[99]
            return np.array([0.0 if g == last else 0.9 for g in ids])
        return _planted(method, k, seed, ids)

    result = cct.analyse(_grid(coverage=coverage), design="nested", reps=REPS)
    step = reference[100].mean - reference[10].mean
    rows = result["scopes"]["ViT-B/32"]["decomposition"]
    mondrian = _row(rows, method="mondrian")
    assert mondrian["selection"]["excludes_zero"] == -1
    assert mondrian["selection_rb"]["mean"] == pytest.approx(
        -step - 0.2 / math.comb(100, 10), abs=1e-12)
    assert mondrian["selection_rb"]["excludes_zero"] == 1
    assert mondrian["selection"]["rb_reproduces"] is False
    weighted = _row(rows, method="weighted")
    assert weighted["selection"]["mean"] == pytest.approx(-0.9 - step)
    assert weighted["selection_rb"]["mean"] == pytest.approx(-0.9 * 0.9 - step)
    assert weighted["selection"]["rb_reproduces"] is True
    assert weighted["same_class"]["excludes_zero"] == 0
    oracle = _row(rows, method="oracle")
    assert oracle["selection"]["excludes_zero"] is None
    assert oracle["selection"]["rb_reproduces"] is None
    assert oracle["selection_rb"]["bonferroni_lo"] is None
    assert mondrian["selection"]["skewness"] is None
    markdown = cct.render_markdown(result)
    line = next(l for l in markdown.splitlines()
                if l.startswith("| Source-Mondrian") and "{" in l and "of 40" in l)
    assert "selection exclusion not reproduced" in line


def test_skewness_of_the_seed_means_is_reported_beside_each_term():
    def coverage(method, k, seed, ids):
        if method == "split" and k == 10:
            return np.full(k, (0.5, 0.5, 0.5, 0.5, 0.9)[seed])
        return np.full(k, 0.4)

    result = cct.analyse(_grid(coverage=coverage, n_seeds=5), design="nested", reps=REPS)
    row = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="split")
    seed_means = 0.4 - np.array([0.5, 0.5, 0.5, 0.5, 0.9])
    assert row["same_class"]["skewness"] == pytest.approx(
        scipy_stats.skew(seed_means, bias=False))
    assert row["same_class"]["skewness"] < -1.0


def _reading_row(*, in_family=True, marked=False, sign=0, free_mean=None, rb=None):
    term = {"marked": marked, "excludes_zero": sign if in_family else None,
            "free_mean": free_mean}
    return {"in_family": in_family, "same_class": dict(term),
            "selection": {**term, "rb_reproduces": rb}}


@pytest.mark.parametrize("row, term, reading", [
    (_reading_row(in_family=False, sign=-1), "selection", "outside"),
    (_reading_row(marked=True, sign=-1, free_mean=-0.2, rb=True), "selection", "bound"),
    (_reading_row(sign=0, free_mean=-0.2), "same_class", "contains"),
    (_reading_row(sign=-1, free_mean=-0.2, rb=True), "selection", "negative"),
    (_reading_row(sign=1, free_mean=0.2, rb=True), "selection", "positive"),
    (_reading_row(sign=1, free_mean=-0.04, rb=True), "selection", "withheld"),
    (_reading_row(sign=-1, free_mean=0.04), "same_class", "withheld"),
    (_reading_row(sign=-1, free_mean=-0.2, rb=False), "selection", "withheld"),
    (_reading_row(sign=-1, free_mean=-0.2, rb=False), "same_class", "negative"),
    (_reading_row(sign=1, free_mean=None, rb=True), "selection", "positive"),
    (_reading_row(sign=1, free_mean=-1e-13, rb=True), "selection", "positive"),
])
def test_term_reading_applies_the_rules_in_their_order(row, term, reading):
    assert cct.term_reading(row, term) == reading
    assert reading in cct.READINGS


def test_term_reading_refuses_a_name_that_is_not_a_term():
    with pytest.raises(ValueError, match="not a term"):
        cct.term_reading(_reading_row(), "total")


def test_term_reading_refuses_a_family_row_without_a_bonferroni_interval():
    """A sign of None in a family row is a damaged file, not an interval that holds zero."""
    row = _reading_row(sign=0)
    row["same_class"]["excludes_zero"] = None
    with pytest.raises(ValueError, match="no Bonferroni interval"):
        cct.term_reading(row, "same_class")


def test_term_reading_agrees_with_the_marks_of_an_analysed_grid(reference):
    """On a grid with a floored method the mark of the report and the reading agree, and a method
    outside the family is read as outside."""
    def coverage(method, k, seed, ids):
        if method == "split":
            return np.zeros(k)
        return np.full(k, 0.8)

    result = cct.analyse(_grid(coverage=coverage, n_seeds=5), design="nested", reps=REPS)
    rows = result["scopes"]["ViT-B/32"]["decomposition"]
    split = _row(rows, method="split")
    assert split["same_class"]["marked"] and split["selection"]["marked"]
    assert {cct.term_reading(split, t) for t in cct.FAMILY_TERMS} == {"bound"}
    assert {cct.term_reading(_row(rows, method="oracle"), t) for t in cct.FAMILY_TERMS} == {"outside"}


def _grid_with_one_planted_cache(planted=("contrast", 2), n_seeds=5):
    def flat(method, k, seed, ids):
        return np.full(k, 0.8)

    def lifted(method, k, seed, ids):
        return np.full(k, 0.5 if method == "split" and k == 10 else 0.8)

    return [
        cct.as_sweep(f"{backbone}/{corruption}_s{severity}",
                     _payload(backbone, corruption, severity, n_seeds=n_seeds,
                              coverage=lifted if (backbone, corruption, severity)
                              == ("ViT-B/32", *planted) else flat))
        for backbone in cct.BACKBONES for corruption in cct.CORRUPTIONS
        for severity in cct.SEVERITIES
    ]


def test_each_term_carries_its_mean_per_cache_in_the_order_of_the_scope():
    """A term planted on one cache shows in that cache's entry and in no other, on the grid scope
    and on the small scope, and the entries average to the term."""
    result = cct.analyse(_grid_with_one_planted_cache(), design="nested", reps=REPS)
    for scope in ("ViT-B/32", cct.MODERATE_SCOPE):
        block = result["scopes"][scope]
        assert "contrast_s2" in block["caches"]
        for row in block["decomposition"]:
            for term in cct.FAMILY_TERMS:
                means = row[term]["caches_means"]
                assert len(means) == block["n_caches"] == len(block["caches"])
                assert np.mean(means) == pytest.approx(row[term]["mean"], abs=1e-12)
        means = _row(block["decomposition"], method="split")["same_class"]["caches_means"]
        by_cache = dict(zip(block["caches"], means))
        assert by_cache.pop("contrast_s2") == pytest.approx(0.3)
        assert all(value == pytest.approx(0.0, abs=1e-12) for value in by_cache.values())
    other = result["scopes"]["ViT-L/14"]["decomposition"]
    assert max(abs(v) for v in _row(other, method="split")["same_class"]["caches_means"]) < 1e-12


def test_the_report_gives_families_for_a_grid_scope_and_caches_for_a_small_one():
    result = cct.analyse(_grid_with_one_planted_cache(), design="nested", reps=REPS)
    text = cct.render_markdown(result)
    grid_part, rest = text.split("## moderate five", 1)
    small_scope = rest.split("\n## ", 1)[0]
    family_head = "Across the corruption families, as description only"
    cache_head = "The two terms cache by cache, as description only"
    assert grid_part.count(family_head) == 2 and cache_head not in grid_part
    stat = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="split")["same_class"]
    assert (stat["n_families_negative"], stat["family_min"]) == (0, pytest.approx(0.0, abs=1e-12))
    assert stat["family_max"] == pytest.approx(0.3 / len(cct.SEVERITIES))
    assert "| Split | 0 of 8, 0.000 to +0.060 |" in grid_part
    assert small_scope.count(cache_head) == 1 and family_head not in small_scope
    caches = result["scopes"][cct.MODERATE_SCOPE]["caches"]
    assert "| method | " + " | ".join(caches) + " |" in small_scope
    by_cache_table = small_scope.split(cache_head, 1)[1]
    split_line = next(line for line in by_cache_table.splitlines() if line.startswith("| Split |"))
    cells = [c.strip() for c in split_line.strip("|").split("|")][1:]
    assert cells[caches.index("contrast_s2")].startswith("+0.300 and ")
    assert all(c.startswith("0.000 and ") for i, c in enumerate(cells)
               if i != caches.index("contrast_s2"))


def test_the_no_shift_terms_carry_their_mean_per_view():
    no_shift = [
        cct.as_sweep(f"noshift_{backbone}_{v}",
                     _payload(backbone, "contrast", 3, view="no_shift", view_seed=v))
        for backbone in cct.BACKBONES for v in range(5)
    ]
    result = cct.analyse(_grid(), design="nested", no_shift=no_shift, reps=REPS)
    block = result["no_shift"]["ViT-B/32"]
    for row in block["decomposition"]:
        for term in cct.FAMILY_TERMS:
            assert len(row[term]["views_means"]) == block["n_views"] == 5
            assert "caches_means" not in row[term]


def test_independent_design_has_no_decomposition_no_family_and_both_bands():
    result = cct.analyse(_grid(design="independent", n_seeds=5), design="independent", reps=REPS)
    assert result["config"]["family"] is None
    scope = result["scopes"]["ViT-B/32"]
    assert scope["decomposition"] is None
    row = _row(scope["per_k"], method="oracle", k=100)
    assert row["band_seed_mean"]["half_width"] == pytest.approx(
        row["band_single_draw"]["half_width"] / np.sqrt(5))
    assert "band_seed_mean" not in result["scopes"][cct.MODERATE_SCOPE]["per_k"][0]


def test_moderate_caches_are_listed_one_by_one_with_means_only():
    result = cct.analyse(_grid(), design="nested", reps=REPS)
    names = [entry["cache"] for entry in result["moderate_caches"]]
    assert names == ["contrast_s1", "contrast_s2", "contrast_s3", "brightness_s1",
                     "brightness_s3"]
    assert result["scopes"][cct.MODERATE_SCOPE]["caches"] == names
    row = _row(result["moderate_caches"][0]["per_k"], method="split", k=10)
    assert row == {"method": "split", "k": 10, "mean": pytest.approx(0.9), "zero_share": 0.0}


def test_no_shift_yardstick_is_reported_beside_its_base_cache_under_shift(reference):
    def shifted(method, k, seed, ids):
        return np.full(k, 0.2 if method == "split" else 0.9)

    def unshifted(method, k, seed, ids):
        return np.full(k, 0.5 if method == "split" else 0.9)

    no_shift = [
        cct.as_sweep(f"noshift_{backbone}_{v}",
                     _payload(backbone, "contrast", 3, coverage=unshifted, view="no_shift",
                              view_seed=v))
        for backbone in cct.BACKBONES for v in range(5)
    ]
    result = cct.analyse(_grid(coverage=shifted), design="nested", no_shift=no_shift, reps=REPS)
    block = result["no_shift"]["ViT-B/32"]
    assert block["n_views"] == 5 and block["view_seeds"] == [0, 1, 2, 3, 4]
    assert block["base_cache"] == "contrast_s3"
    row = _row(block["per_k"], method="split", k=100)
    assert row["mean"] == pytest.approx(0.5)
    assert row["diff"] == pytest.approx(0.5 - reference[100].mean)
    assert row["base_cache_shift_mean"] == pytest.approx(0.2)


def test_no_shift_is_absent_when_no_file_is_given():
    result = cct.analyse(_grid(), design="nested", reps=REPS)
    assert result["no_shift"] is None
    row = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="split")
    assert "no_shift" not in row["selection"] and "minus_no_shift" not in row["selection"]
    assert "shift minus no shift" not in cct.render_markdown(result)


def _views(coverage, n_seeds=3, n_views=5):
    return [cct.as_sweep(f"noshift_{backbone}_{v}",
                         _payload(backbone, "contrast", 3, coverage=coverage, view="no_shift",
                                  view_seed=v, n_seeds=n_seeds))
            for backbone in cct.BACKBONES for v in range(n_views)]


def test_each_term_is_given_beside_its_value_without_shift(reference):
    """Split targets marginal coverage, so its selection term is negative with no shift at all."""
    def gap(low):
        def coverage(method, k, seed, ids):
            if method != "split":
                return np.full(k, 0.9)
            kept = set(_order(seed)[:10])
            return np.array([0.6 if g in kept else low for g in ids])
        return coverage

    result = cct.analyse(_grid(coverage=gap(0.3)), design="nested", no_shift=_views(gap(0.4)),
                         reps=REPS)
    step = reference[100].mean - reference[10].mean
    for name in ("ViT-B/32", "ViT-L/14", cct.MODERATE_SCOPE):
        row = _row(result["scopes"][name]["decomposition"], method="split")
        assert row["selection"]["mean"] == pytest.approx(-0.3 - step)
        assert row["selection"]["no_shift"]["mean"] == pytest.approx(-0.2 - step)
        paired = row["selection"]["minus_no_shift"]
        assert paired["mean"] == pytest.approx(-0.1)
        assert (paired["ci_lo"], paired["ci_hi"]) == pytest.approx((-0.1, -0.1))
        assert row["same_class"]["minus_no_shift"]["mean"] == pytest.approx(0.0, abs=1e-12)
    block = result["no_shift"]["ViT-L/14"]
    view_row = _row(block["decomposition"], method="split")
    assert view_row["in_family"] is False and view_row["selection"]["bonferroni_lo"] is None
    assert view_row["selection"]["mean"] == pytest.approx(-0.2 - step)
    assert view_row["selection"]["n_views"] == 5 == view_row["selection"]["n_views_negative"]
    assert "n_caches" not in view_row["selection"]
    assert view_row["limits"]["worst_small"] == {"at_zero": 0.0, "at_one": 0.0}
    markdown = cct.render_markdown(result)
    for needle in ("selection, shift minus no shift", "without shift, each with its unadjusted"):
        assert needle in markdown, needle


def test_the_paired_difference_from_no_shift_uses_the_seeds_as_replicates():
    def shifted(method, k, seed, ids):
        return np.full(k, 0.5 + 0.1 * seed if k == 10 else 0.5)

    def unshifted(method, k, seed, ids):
        return np.full(k, 0.5 + 0.02 * seed if k == 10 else 0.5)

    result = cct.analyse(_grid(coverage=shifted, n_seeds=4), design="nested",
                         no_shift=_views(unshifted, n_seeds=4), reps=REPS)
    stat = _row(result["scopes"]["ViT-B/32"]["decomposition"], method="mondrian")["same_class"]
    differences = -(0.1 - 0.02) * np.arange(4)
    assert stat["minus_no_shift"]["mean"] == pytest.approx(differences.mean())
    assert (stat["minus_no_shift"]["ci_lo"], stat["minus_no_shift"]["ci_hi"]) == pytest.approx(
        cct.seed_interval(differences, 0.95))


def test_no_shift_views_must_match_the_sweep_in_design_and_seeds():
    with pytest.raises(ValueError, match="subset_seeds"):
        cct.analyse(_grid(n_seeds=3), design="nested", no_shift=_views(None, n_seeds=2),
                    reps=REPS)
    with pytest.raises(ValueError, match="nested design only"):
        cct.analyse(_grid(design="independent", n_seeds=5), design="independent",
                    no_shift=_views(None, n_seeds=5), reps=REPS)


def _verify(payload, **kwargs):
    kwargs.setdefault("design", "nested")
    kwargs.setdefault("view", "shift")
    return cct.verify_payload(payload, "file", **kwargs)


def test_a_well_formed_payload_verifies():
    assert _verify(_payload("ViT-B/32", "fog", 2)) == []
    assert _verify(_payload("ViT-B/32", "fog", 2, design="independent"),
                   design="independent") == []
    assert _verify(_payload("ViT-L/14", "contrast", 3, view="no_shift", view_seed=4),
                   view="no_shift") == []


@pytest.mark.parametrize("damage, phrase", [
    (lambda p: p["config"].update(subset_design="independent"), "subset design"),
    (lambda p: p["config"].update(view="no_shift"), "view"),
    (lambda p: p["config"].update(alpha=0.05), "alpha"),
    (lambda p: p["config"].update(n_cal_per_class=25), "n_cal_per_class"),
    (lambda p: p["config"].update(k_values=[10, 100]), "k_values"),
    (lambda p: p["config"].update(cache="caches/cifar100c_mild.npz"), "cache"),
    (lambda p: p["records"].pop(), "missing"),
    (lambda p: p["records"].append(dict(p["records"][0])), "duplicate"),
    (lambda p: p["records"][0].update(n_shortfall_classes=2), "shortfall"),
    (lambda p: p["records"][0].update(n_eval_total=1), "n_eval_total"),
    (lambda p: p["records"][0].update(per_class_coverage=[None] * 10), "per_class_coverage"),
    (lambda p: p["records"][0].update(worst_class=0.123), "worst_class"),
    (lambda p: p["class_orders"].pop(), "class_orders"),
    (lambda p: p["class_orders"][0].__setitem__(0, p["class_orders"][0][1]), "class_orders"),
    (lambda p: p["provenance"].update(code_version=None), "code_version"),
    (lambda p: p["provenance"].update(code_version="unknown"), "code_version"),
    (lambda p: p.pop("provenance"), "provenance"),
    (lambda p: p["config"].pop("method_seed"), "method_seed"),
    (lambda p: p["problems"].pop(), "problems block"),
    (lambda p: p.pop("problems"), "problems block"),
    (lambda p: p["problems"][0]["class_ids"].reverse(), "class_ids"),
    (lambda p: p["problems"][0]["n_eval_per_class"].__setitem__(slice(0, 2), [49, 51]),
     "evaluation rows for every class"),
    (lambda p: p["problems"][0].update(eval_top1_per_class=[0.5]), "eval_top1_per_class"),
    (lambda p: p["records"][0].pop("n_cal_per_class"), "n_cal_per_class"),
    (lambda p: next(r for r in p["records"] if r["method"] == "mondrian")
     ["n_cal_per_class"].__setitem__(slice(0, 2), [49, 51]), "calibration points for every class"),
    (lambda p: next(r for r in p["records"] if r["method"] == "oracle")
     ["n_cal_per_class"].__setitem__(0, 25), "calibration points for every class"),
    (lambda p: p["records"][0]["per_class_coverage"].__setitem__(3, float("nan")),
     "per_class_coverage"),
    (lambda p: p["records"][0]["per_class_coverage"].__setitem__(3, 1.5), "per_class_coverage"),
    (lambda p: p["records"][0].update(worst_class=float("nan")), "worst_class"),
    (lambda p: p["records"][0].update(set_over_k=float("nan")), "set_over_k"),
    (lambda p: p["records"][0].pop("set_over_k"), "set_over_k"),
    (lambda p: p["config"].pop("acp_tie_break"), "acp_tie_break"),
    (lambda p: p["config"].update(acp_tie_break="index"), "acp_tie_break"),
    (lambda p: p["provenance"].pop("numpy"), "provenance.numpy"),
    (lambda p: p["provenance"].update(python=""), "provenance.python"),
    (lambda p: p["provenance"].update(sklearn=None), "provenance.sklearn"),
    (lambda p: next(f for f in p["problems"] if f["k"] == 100 and f["subset_seed"] == 0)
     .update(eval_top1_per_class=[0.9] * 100), "top-1 accuracy rises with K"),
])
def test_a_damaged_payload_is_named(damage, phrase):
    payload = _payload("ViT-B/32", "fog", 2)
    damage(payload)
    problems = _verify(payload)
    assert problems and any(phrase in problem for problem in problems), problems


def test_a_nan_survives_the_json_round_trip_and_is_still_refused(tmp_path):
    payload = _payload("ViT-B/32", "fog", 2)
    payload["records"][5]["per_class_coverage"][7] = float("nan")
    path = tmp_path / "nested_k_sweep_fog_s2.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)
    sweeps, problems = cct.load_sweeps([path], design="nested", view="shift")
    assert sweeps == [] and any("per_class_coverage" in problem for problem in problems)


def test_the_tie_rule_is_asked_only_of_a_nested_file_that_holds_acp():
    without = [m for m in ALL_METHODS if m != "acp"]
    payload = _payload("ViT-B/32", "fog", 2)
    payload["records"] = [r for r in payload["records"] if r["method"] != "acp"]
    del payload["config"]["acp_tie_break"]
    assert _verify(payload, methods=without) == []
    released = _payload("ViT-B/32", "fog", 2, design="independent")
    assert "acp_tie_break" not in released["config"]
    assert _verify(released, design="independent") == []


def test_a_reversed_per_class_layout_is_caught_by_the_accuracies():
    """Top-1 accuracy by class that falls with K in the right layout."""
    payload = _payload("ViT-B/32", "fog", 2, n_seeds=1)
    by_k = {f["k"]: f for f in payload["problems"]}
    for k, fact in by_k.items():
        fact["eval_top1_per_class"] = [round(0.9 - 0.004 * g - 0.001 * k, 6)
                                       for g in fact["class_ids"]]
    assert _verify(payload) == []
    by_k[10]["eval_top1_per_class"].reverse()
    assert any("top-1 accuracy rises with K" in problem for problem in _verify(payload))


def test_a_wrong_seed_count_is_named():
    problems = _verify(_payload("ViT-B/32", "fog", 2, n_seeds=3), n_seeds=30)
    assert any("subset_seeds" in problem for problem in problems)


def test_a_nested_file_is_refused_as_independent_and_the_other_way_round():
    assert _verify(_payload("ViT-B/32", "fog", 2), design="independent")
    assert _verify(_payload("ViT-B/32", "fog", 2, design="independent"), design="nested")


def test_grid_problems_name_a_missing_and_a_repeated_cache():
    sweeps = _grid()
    assert cct.grid_problems(sweeps, cct.BACKBONES) == []
    missing = [s for s in sweeps if s.key != ("ViT-L/14", "fog", 3)]
    problems = cct.grid_problems(missing, cct.BACKBONES)
    assert len(problems) == 1 and "vitL_fog_s3" in problems[0]
    problems = cct.grid_problems(sweeps + [sweeps[0]], cct.BACKBONES)
    assert any("more than one file" in problem for problem in problems)
    problems = cct.grid_problems(_grid(backbones=("ViT-B/32",)))
    assert len(problems) == 1 and "40" in problems[0]


def _write_grid(directory: Path, *, n_seeds=2, skip=(), design="nested") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for backbone in cct.BACKBONES:
        for corruption in cct.CORRUPTIONS:
            for severity in cct.SEVERITIES:
                if (backbone, corruption, severity) in skip:
                    continue
                middle = "vitL_" if backbone == "ViT-L/14" else ""
                payload = _payload(backbone, corruption, severity, n_seeds=n_seeds, design=design)
                path = directory / f"nested_k_sweep_{middle}{corruption}_s{severity}.json.gz"
                with gzip.open(path, "wt") as handle:
                    json.dump(payload, handle)
    for backbone, tag in (("ViT-B/32", "vitB"), ("ViT-L/14", "vitL")):
        for v in range(5):
            payload = _payload(backbone, "contrast", 3, n_seeds=n_seeds, view="no_shift",
                               view_seed=v)
            with gzip.open(directory / f"noshift_k_sweep_{tag}_v{v}.json.gz", "wt") as handle:
                json.dump(payload, handle)


def _verify_args(directory: Path, *extra: str) -> list[str]:
    return ["--design", "nested", "--inputs-glob", str(directory / "nested_k_sweep_*.json.gz"),
            "--no-shift-glob", str(directory / "noshift_k_sweep_*.json.gz"),
            "--expect-seeds", "2", "--expect-no-shift", "10", "--verify-only", *extra]


def test_verify_only_accepts_a_complete_set(tmp_path, capsys):
    _write_grid(tmp_path / "out")
    assert cct.main(_verify_args(tmp_path / "out")) == 0
    out = capsys.readouterr().out
    assert "VERIFY OK" in out and "80 sweep files" in out and "10 no-shift files" in out
    assert "abc1234" in out
    assert "ViT-B/32: 1 distinct source set" in out
    assert not list(tmp_path.glob("*.json")) and not list(tmp_path.glob("*.md"))


def test_verify_only_refuses_a_missing_cache_a_truncated_file_and_a_wrong_seed_count(tmp_path, capsys):
    out_dir = tmp_path / "out"
    _write_grid(out_dir, skip={("ViT-B/32", "fog", 3)})
    assert cct.main(_verify_args(out_dir)) == 1
    assert "fog_s3" in capsys.readouterr().out

    _write_grid(out_dir)
    victim = out_dir / "nested_k_sweep_contrast_s1.json.gz"
    victim.write_bytes(victim.read_bytes()[:200])
    assert cct.main(_verify_args(out_dir)) == 1
    text = capsys.readouterr().out
    assert "nested_k_sweep_contrast_s1.json.gz" in text and "unreadable" in text
    assert "VERIFY FAILED" in text

    _write_grid(out_dir)
    args = _verify_args(out_dir)
    args[args.index("--expect-seeds") + 1] = "30"
    assert cct.main(args) == 1
    assert "subset_seeds" in capsys.readouterr().out

    (out_dir / "noshift_k_sweep_vitL_v4.json.gz").unlink()
    assert cct.main(_verify_args(out_dir)) == 1
    assert "no-shift" in capsys.readouterr().out


def test_files_of_two_code_versions_are_not_one_run(tmp_path, capsys):
    """A resubmission after a code fix must not leave files of two versions that verify together."""
    out_dir = tmp_path / "out"
    _write_grid(out_dir)
    victim = out_dir / "nested_k_sweep_vitL_fog_s3.json.gz"
    payload = _payload("ViT-L/14", "fog", 3, n_seeds=2, stamp="fff9999")
    with gzip.open(victim, "wt") as handle:
        json.dump(payload, handle)
    assert cct.main(_verify_args(out_dir)) == 1
    text = capsys.readouterr().out
    assert "2 code versions (abc1234, fff9999)" in text and "VERIFY FAILED" in text


def test_no_shift_files_must_be_distinct_view_seeds_of_one_cache(tmp_path, capsys):
    out_dir = tmp_path / "out"
    _write_grid(out_dir)
    with gzip.open(out_dir / "noshift_k_sweep_vitB_v4.json.gz", "wt") as handle:
        json.dump(_payload("ViT-B/32", "contrast", 3, n_seeds=2, view="no_shift", view_seed=0),
                  handle)
    assert cct.main(_verify_args(out_dir)) == 1
    assert "repeat a view seed" in capsys.readouterr().out

    _write_grid(out_dir)
    with gzip.open(out_dir / "noshift_k_sweep_vitL_v4.json.gz", "wt") as handle:
        json.dump(_payload("ViT-L/14", "fog", 1, n_seeds=2, view="no_shift", view_seed=4), handle)
    assert cct.main(_verify_args(out_dir)) == 1
    assert "come from 2 caches" in capsys.readouterr().out


def test_no_shift_files_must_split_evenly_and_come_from_the_base_cache(tmp_path, capsys):
    """Ten files that are six views of one backbone and four of the other pass a count of ten,
    and so do ten views of another cache."""
    out_dir = tmp_path / "out"
    _write_grid(out_dir)
    (out_dir / "noshift_k_sweep_vitL_v4.json.gz").unlink()
    with gzip.open(out_dir / "noshift_k_sweep_vitB_v5.json.gz", "wt") as handle:
        json.dump(_payload("ViT-B/32", "contrast", 3, n_seeds=2, view="no_shift", view_seed=5),
                  handle)
    assert cct.main(_verify_args(out_dir)) == 1
    text = capsys.readouterr().out
    assert "not the same number for each backbone (6 of ViT-B/32, 4 of ViT-L/14)" in text

    other = tmp_path / "other"
    _write_grid(other)
    for backbone, tag in (("ViT-B/32", "vitB"), ("ViT-L/14", "vitL")):
        for v in range(5):
            with gzip.open(other / f"noshift_k_sweep_{tag}_v{v}.json.gz", "wt") as handle:
                json.dump(_payload(backbone, "fog", 1, n_seeds=2, view="no_shift", view_seed=v),
                          handle)
    assert cct.main(_verify_args(other)) == 1
    text = capsys.readouterr().out
    assert "the no-shift view is drawn from contrast_s3 for both backbones" in text
    assert "ViT-B/32: a no-shift file is a view of fog_s1" in text
    assert "ViT-L/14: a no-shift file is a view of vitL_fog_s1" in text


def test_files_of_two_library_versions_are_not_one_run(tmp_path, capsys):
    out_dir = tmp_path / "out"
    _write_grid(out_dir)
    victim = out_dir / "nested_k_sweep_vitL_fog_s3.json.gz"
    payload = _payload("ViT-L/14", "fog", 3, n_seeds=2)
    payload["provenance"]["numpy"] = "2.0.0"
    with gzip.open(victim, "wt") as handle:
        json.dump(payload, handle)
    assert cct.main(_verify_args(out_dir)) == 1
    text = capsys.readouterr().out
    assert "2 versions of numpy (1.26.4, 2.0.0)" in text and "VERIFY FAILED" in text


def test_verify_only_refuses_a_file_written_before_the_tie_fix(tmp_path, capsys):
    """Every nested file must name the tie rule, so a file written before the fix is refused."""
    out_dir = tmp_path / "out"
    _write_grid(out_dir)
    victim = out_dir / "noshift_k_sweep_vitB_v2.json.gz"
    payload = _payload("ViT-B/32", "contrast", 3, n_seeds=2, view="no_shift", view_seed=2)
    del payload["config"]["acp_tie_break"]
    with gzip.open(victim, "wt") as handle:
        json.dump(payload, handle)
    assert cct.main(_verify_args(out_dir)) == 1
    text = capsys.readouterr().out
    assert "noshift_k_sweep_vitB_v2.json.gz: acp_tie_break is None, expected 'random'" in text


def test_verify_only_refuses_an_empty_glob(tmp_path, capsys):
    assert cct.main(_verify_args(tmp_path / "nothing")) == 1
    assert "no file matches" in capsys.readouterr().out


def test_the_command_writes_json_and_markdown(tmp_path, capsys):
    _write_grid(tmp_path / "out")
    out_json, out_md = tmp_path / "trend.json", tmp_path / "TREND.md"
    args = ["--design", "nested", "--inputs-glob", str(tmp_path / "out" / "nested_k_sweep_*.json.gz"),
            "--no-shift-glob", str(tmp_path / "out" / "noshift_k_sweep_*.json.gz"),
            "--out-json", str(out_json), "--out-md", str(out_md), "--reps", str(REPS)]
    assert cct.main(args) == 0
    capsys.readouterr()
    result = json.loads(out_json.read_text())
    assert result["config"]["design"] == "nested"
    assert result["config"]["n_files"] == 80
    assert result["config"]["code_versions"] == ["abc1234"]
    assert result["config"]["distinct_source_sets"] == {"ViT-B/32": 1, "ViT-L/14": 1}
    assert result["config"]["acp_tie_breaks"] == ["random"]
    assert result["config"]["family"]["bound_share"] == cct.BOUND_SHARE == 0.5
    markdown = out_md.read_text()
    for needle in ("same-class term", "selection term", "one image set", "t interval",
                   "The family holds 20 intervals", "No-shift yardstick", "contrast_s3",
                   cct.MODERATE_SCOPE, "selected on outcomes",
                   "unadjusted and conditional on the image set", "Limits of the coverage range",
                   "Checks on the intervals", "selection, shift minus no shift",
                   "less than about 0.02 is not read as evidence",
                   "Split, weighted, clustered and ACP calibrate one threshold",
                   "the rows in those positions change with the seed",
                   "Weighted fits its domain classifier on the K kept logits"):
        assert needle in markdown, needle
    assert "index rule" not in markdown
    assert "bootstrap" not in markdown
    for forbidden in ("—", "–", " - "):
        assert forbidden not in markdown, forbidden
    assert re.search(r"\bnan\b", markdown, re.IGNORECASE) is None
    first = out_json.read_text(), out_md.read_text()
    assert cct.main(args) == 0
    capsys.readouterr()
    assert (out_json.read_text(), out_md.read_text()) == first


def test_the_command_refuses_to_analyse_a_set_that_does_not_verify(tmp_path, capsys):
    _write_grid(tmp_path / "out", skip={("ViT-L/14", "contrast", 1)})
    out_json, out_md = tmp_path / "trend.json", tmp_path / "TREND.md"
    args = ["--design", "nested", "--inputs-glob", str(tmp_path / "out" / "nested_k_sweep_*.json.gz"),
            "--out-json", str(out_json), "--out-md", str(out_md), "--reps", str(REPS)]
    assert cct.main(args) == 1
    assert "VERIFY FAILED" in capsys.readouterr().out
    assert not out_json.exists() and not out_md.exists()


def _committed_files() -> list[Path]:
    return sorted(_ANALYSIS.glob("controlled_k_sweep_*.json"))


@pytest.fixture(scope="module")
def committed_result():
    files = _committed_files()
    if len(files) != 80:
        pytest.skip("the 80 committed every-cache sweep files are not in this checkout")
    sweeps, problems = cct.load_sweeps(files, design="independent", view="shift")
    assert problems == []
    return cct.analyse(sweeps, design="independent")


def test_committed_files_reproduce_the_registry_counts_at_k_100(committed_result):
    """The registry counts, per method, how many of the 80 caches read below, within and above
    the reference band at K 100."""
    rn = _load_script(_ROOT / "scripts" / "number_registry.py", "_trend_number_registry")
    registry = {entry.name: entry.value for entry in rn.ksweep_all_entries(_ROOT)}
    assert registry["ksweep_all.n_caches"] == "80"
    for method in ALL_METHODS:
        for side in ("below", "within", "above"):
            total = sum(
                _row(committed_result["scopes"][backbone]["per_k"], method=method, k=100)[
                    "band_seed_mean"][side]
                for backbone in cct.BACKBONES)
            assert str(total) == registry[f"ksweep_all.K100.{method}.n_{side}"], (method, side)


def test_committed_files_were_all_written_under_one_acp_tie_rule(committed_result):
    """The committed K-sweep files record one ACP tie rule, and the report names the index rule
    exactly when that is the rule."""
    rules = committed_result["config"]["acp_tie_breaks"]
    assert rules in (["index"], ["random"]), rules
    markdown = cct.render_markdown(committed_result)
    assert ("index rule" in markdown) is (rules == ["index"])


def test_committed_files_put_the_oracle_on_the_reference_at_every_k(committed_result):
    for backbone in cct.BACKBONES:
        for k in cct.K_VALUES:
            row = _row(committed_result["scopes"][backbone]["per_k"], method="oracle", k=k)
            assert abs(row["diff"]) < 0.02, (backbone, k, row["diff"])
            assert row["zero_share"] == 0.0


def test_the_real_sweep_output_feeds_the_analysis(tmp_path, capsys):
    """Contract between the two scripts: a nested file written by scripts/controlled_k_sweep.py
    verifies, and its class orders line up with its per-class vectors, so the K 10 worst class is
    recovered from the K 100 layout."""
    from pcpbench.datasets.synthetic import make_synthetic_shift

    cks = _load_script(_ROOT / "scripts" / "controlled_k_sweep.py", "_trend_controlled_k_sweep")
    cache = tmp_path / "cifar100c_contrast_s3.npz"
    make_synthetic_shift(K=100, n_per_class=100, shift=0.6, seed=0).to_npz(cache)
    out = tmp_path / "nested_k_sweep_contrast_s3.json.gz"
    assert cks.main(["--cache", str(cache), "--k-values", "10,25,50,100", "--subset-seeds", "2",
                     "--n-cal-per-class", "50", "--subset-design", "nested", "--stamp", "abc1234",
                     "--out", str(out)]) == 0
    capsys.readouterr()
    payload = cct.load_payload(out)
    assert cct.verify_payload(payload, out.name, design="nested", view="shift", n_seeds=2) == []
    assert payload["config"]["acp_tie_break"] == cct.ACP_TIE_BREAK
    assert all(payload["provenance"][library] for library in cct.LIBRARY_KEYS)
    sweep = cct.as_sweep(out.name, payload)
    for seed in range(2):
        for method in ("mondrian", "oracle"):
            small = cct.class_coverage(sweep, method, 10, seed)
            assert np.count_nonzero(~np.isnan(small)) == 10
            assert np.nanmin(small) == pytest.approx(sweep.cells[(method, 10, seed)]["worst_class"])
            kept = np.asarray(sweep.class_orders[seed][:10])
            assert not np.isnan(small[kept]).any()
            terms = cct.decomposition_terms(sweep, method, seed, 10, 100, {
                10: cct.Reference(0.8, 0.05), 100: cct.Reference(0.7, 0.04)})
            assert terms.worst_large <= terms.kept_min_large
            assert terms.same_class + terms.selection == pytest.approx(terms.total)
            assert terms.selection_rb <= 0.1 + 1e-12

    facts = {(f["k"], f["subset_seed"]): f for f in payload["problems"]}
    small, larger = facts[(10, 0)], facts[(25, 0)]
    at_25 = dict(zip(larger["class_ids"], larger["eval_top1_per_class"]))
    reversed_small = small["eval_top1_per_class"][::-1]
    expected = any(at_25[g] > accuracy + 1e-12
                   for g, accuracy in zip(small["class_ids"], reversed_small))
    small["eval_top1_per_class"] = reversed_small
    found = any("top-1 accuracy rises with K" in problem for problem in cct.verify_payload(
        payload, out.name, design="nested", view="shift", n_seeds=2))
    assert found is expected

    wide = tmp_path / "wide" / "cifar100c_contrast_s3.npz"
    wide.parent.mkdir()
    make_synthetic_shift(K=100, n_per_class=160, shift=0.6, seed=0).to_npz(wide)
    view = tmp_path / "noshift_k_sweep_vitB_v1.json.gz"
    assert cks.main(["--cache", str(wide), "--k-values", "10,25,50,100", "--subset-seeds", "2",
                     "--n-cal-per-class", "50", "--subset-design", "nested", "--view", "no_shift",
                     "--view-seed", "1", "--stamp", "abc1234", "--out", str(view)]) == 0
    capsys.readouterr()
    assert cct.verify_payload(cct.load_payload(view), view.name, design="nested",
                              view="no_shift", n_seeds=2) == []
