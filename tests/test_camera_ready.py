"""The numbers and tables of the camera-ready paper, checked against the committed artifacts."""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from types import ModuleType

import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS = _ROOT / "outputs" / "analysis"

K_VALUES = (10, 25, 50, 100)
MODERATE = "moderate five (ViT-B/32)"
BACKBONES = ("ViT-B/32", "ViT-L/14")


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cr() -> ModuleType:
    return _load_script(_ROOT / "scripts" / "camera_ready_numbers.py", "cr_numbers_under_test")


@pytest.fixture(scope="module")
def pool(cr: ModuleType) -> dict[str, str]:
    return cr.build_pool(_ROOT)


@pytest.fixture(scope="module")
def trend() -> dict:
    return json.loads((_ANALYSIS / "all_cache_trend.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nested() -> dict:
    return json.loads((_ANALYSIS / "nested_trend.json").read_text(encoding="utf-8"))


def _rows(trend: dict, scope: str) -> dict[tuple[str, int], dict]:
    return {(r["method"], r["k"]): r for r in trend["scopes"][scope]["per_k"]}


def _reference(trend: dict) -> dict[int, float]:
    return {int(k): v["mean"] for k, v in trend["reference"].items()}


def _num(pool: dict[str, str], key: str) -> float:
    return float(pool[key])


def _row_file(dataset: str, method: str) -> dict:
    return json.loads((_ROOT / "outputs" / f"{dataset}_{method}.json").read_text(encoding="utf-8"))


def _set_share(row: dict) -> float:
    return row["mean_set_size"] / row["n_classes"]


def _guard_demotes(row: dict) -> bool:
    share = _set_share(row)
    return share >= 0.90 or (share >= 0.60 and row["marginal"] >= 0.95)
CORRUPTIONS = ("brightness", "contrast", "defocus_blur", "fog", "gaussian_blur", "gaussian_noise",
               "jpeg_compression", "shot_noise")


def _informative_pairs(method: str, backbone_tag: str) -> list[tuple[dict, dict]]:
    def saturated(row: dict) -> bool:
        return float(f"{row['worst_class']:.3f}") >= 0.99 and float(f"{_set_share(row):.2f}") >= 0.90

    pairs = [(_row_file(f"cifar10c_{backbone_tag}{corruption}_s{severity}", method),
              _row_file(f"cifar100c_{backbone_tag}{corruption}_s{severity}", method))
             for corruption in CORRUPTIONS for severity in range(1, 6)]
    assert len(pairs) == 40
    return [(low, high) for low, high in pairs if not saturated(low) and not saturated(high)]


def _mean_change(pairs: list[tuple[dict, dict]]) -> float:
    return mean(float(f"{high['worst_class']:.3f}") - float(f"{low['worst_class']:.3f}")
                for low, high in pairs)


def test_generated_files_equal_a_fresh_run(cr: ModuleType, capsys: pytest.CaptureFixture) -> None:
    code = cr.main(["--root", str(_ROOT), "--check"])
    out = capsys.readouterr().out
    assert code == 0, out


def _note_pattern(note: str) -> re.Pattern:
    parts = []
    for segment in note.split("."):
        if segment == "*":
            parts.append(r"[^\s]+")
        elif "/" in segment:
            parts.append("(?:" + "|".join(re.escape(s) for s in segment.split("/")) + ")")
        else:
            parts.append(re.sub(r"<[^>]+>", "[^.]+", re.escape(segment).replace(r"\<", "<")
                                .replace(r"\>", ">")))
    return re.compile(r"\.".join(parts))


def test_the_note_patterns_match_their_keys_and_nothing_wider() -> None:
    assert _note_pattern("derived.label.<method>.m<m>.ci_lo/ci_hi").fullmatch(
        "derived.label.ppi.m10.ci_hi")
    assert not _note_pattern("derived.label.<method>.m<m>.ci_lo/ci_hi").fullmatch(
        "derived.label.ppi.m10.marginal")
    assert _note_pattern("derived.delta.<method>.*").fullmatch("derived.delta.acp.informative.magnitude")
    assert not _note_pattern("derived.delta.<method>.*").fullmatch("derived.delta.acp")
    assert not _note_pattern("derived.guarantee.n50").fullmatch("derived.guarantee.n51")


def test_every_printed_derived_key_has_a_documented_formula(cr: ModuleType) -> None:
    patterns = [_note_pattern(note) for note in cr.DERIVED_NOTES]
    undocumented = [k for k in cr.used_keys(_ROOT) if k.startswith("derived.")
                    and not any(p.fullmatch(k) for p in patterns)]
    assert undocumented == []


def test_prose_and_table_print_the_same_controlled_values(cr: ModuleType, pool: dict[str, str],
                                                          trend: dict) -> None:
    """The prose quotes the registry (sweep.per_k), Table tab:perk prints the trend file; both
    must print the same mean, difference and reference."""
    for method in cr.HEADLINE_ORDER:
        for k in K_VALUES:
            assert pool[f"sweep.per_k.{method}.K{k}.mean"] == pool[f"trend.moderate.{method}.K{k}.mean"]
            assert _num(pool, f"sweep.per_k.{method}.K{k}.excess") == \
                _num(pool, f"trend.moderate.{method}.K{k}.diff")
    floor = {r["K"]: r["floor_mean"] for r in
             json.loads((_ANALYSIS / "validity_floor.json").read_text(encoding="utf-8"))["rows"]
             if r["setting"] == "k_sweep"}
    for k, value in _reference(trend).items():
        assert math.isclose(value, floor[k], abs_tol=1e-12)
        assert pool[f"floor.k_sweep.K{k}.floor_mean"] == f"{value:.3f}"


def test_source_mondrian_and_the_oracle_hold_the_reference_without_shift(pool: dict[str, str]) -> None:
    n = int(pool["derived.no_shift.n_distinct"])
    assert int(pool["derived.no_shift.mondrian.n_within_two_sd"]) == n
    assert int(pool["derived.no_shift.oracle.n_within_two_sd"]) == n
    assert int(pool["derived.no_shift.mondrian.n_within_two_se"]) == n
    assert 0 < int(pool["derived.no_shift.oracle.n_within_two_se"]) <= n


def test_split_misses_the_per_class_reference_without_shift(pool: dict[str, str]) -> None:
    n = int(pool["derived.no_shift.n_distinct"])
    assert n / 2 < int(pool["derived.no_shift.split.n_below_two_sd"]) <= n


def test_the_threshold_is_infinite_below_nine_calibration_examples() -> None:
    """The rank ceil((n + 1) * 0.90) exceeds n exactly for n <= 8."""
    from pcpbench.metrics.validity_floor import coverage_law

    assert all(coverage_law(n, 0.10)[1] == 0 for n in range(1, 9))
    assert all(coverage_law(n, 0.10)[1] > 0 for n in range(9, 60))


def test_chance_values_of_the_fraction_undercovered_follow_the_stated_formula(
        pool: dict[str, str]) -> None:
    """F(ceil(0.90 m) - 1), F the BetaBinomial(m, r, n + 1 - r) distribution function with r =
    ceil((n + 1) * 0.90)."""
    for key, n, m in (("cifar10c.K10", 500, 500), ("k_sweep.K100", 50, 50),
                      ("imagenet1k.K1000", 25, 25)):
        r = math.ceil((n + 1) * 0.9 - 1e-12)
        chance = stats.betabinom.cdf(math.ceil(0.9 * m - 1e-12) - 1, m, r, n + 1 - r)
        assert f"{chance:.3f}" == pool[f"floor.{key}.fp_rate_at_0.90"], key


def test_oracle_interval_contains_the_reference_at_every_k(trend: dict) -> None:
    rows, ref = _rows(trend, MODERATE), _reference(trend)
    for k in K_VALUES:
        assert rows[("oracle", k)]["ci_lo"] <= ref[k] <= rows[("oracle", k)]["ci_hi"], k


def test_four_methods_have_their_interval_below_the_reference_at_k100(trend: dict) -> None:
    """Split, weighted, PPI and Source-Mondrian, the methods that calibrate without target labels
    in this sweep."""
    rows, ref = _rows(trend, MODERATE), _reference(trend)
    for method in ("split", "ppi", "weighted", "mondrian"):
        assert rows[(method, 100)]["ci_hi"] < ref[100], method
    below = sorted(("split", "ppi", "weighted", "mondrian"), key=lambda m: rows[(m, 100)]["mean"])
    assert below[0] == "ppi" and below[-1] == "mondrian"


def test_split_and_source_mondrian_are_below_at_every_k(trend: dict) -> None:
    rows, ref = _rows(trend, MODERATE), _reference(trend)
    for method in ("split", "mondrian"):
        for k in K_VALUES:
            assert rows[(method, k)]["ci_hi"] < ref[k], (method, k)


def test_weighted_and_ppi_are_below_from_k25_while_k10_reaches_it(trend: dict) -> None:
    rows, ref = _rows(trend, MODERATE), _reference(trend)
    for method in ("weighted", "ppi"):
        assert rows[(method, 10)]["ci_hi"] >= ref[10], method
        for k in (25, 50, 100):
            assert rows[(method, k)]["ci_hi"] < ref[k], (method, k)


def test_acp_interval_is_above_the_reference_at_every_k_with_sets_of_the_printed_share(
        pool: dict[str, str], trend: dict) -> None:
    """ACP's interval lies above the reference at every K, and its mean set holds a share of the
    label space that runs from the K=10 value to the K=100 value, most of it."""
    rows, ref = _rows(trend, MODERATE), _reference(trend)
    for k in K_VALUES:
        assert rows[("acp", k)]["ci_lo"] > ref[k], k
    shares = [rows[("acp", k)]["set_over_k"] for k in K_VALUES]
    assert min(shares) == shares[0] and max(shares) == shares[-1]
    assert shares[0] > 0.75
    assert pool["trend.moderate.acp.K10.set_over_k"] == f"{shares[0]:.2f}"
    assert pool["trend.moderate.acp.K100.set_over_k"] == f"{shares[-1]:.2f}"
    caches = trend["scopes"][MODERATE]["caches"]
    for k, share in zip(K_VALUES, shares, strict=True):
        raw = mean(r["set_over_k"] for cache in caches for r in _sweep_records()[cache]
                   if r["method"] == "acp" and r["k"] == k)
        assert abs(raw - share) < 1e-9, k


def test_difference_widens_for_four_methods_and_acp_stays_above(pool: dict[str, str],
                                                               trend: dict) -> None:
    rows = _rows(trend, MODERATE)
    for method in ("split", "ppi", "weighted", "mondrian"):
        assert rows[(method, 100)]["diff"] < rows[(method, 10)]["diff"] < 0, method
    for k in K_VALUES:
        assert rows[("acp", k)]["diff"] > 0, k
    for k in (10, 100):
        assert abs(_num(pool, f"sweep.per_k.acp.K{k}.excess") - rows[("acp", k)]["diff"]) < 1e-3, k


def test_the_five_caches_have_label_free_coverage_above_zero_at_k10(trend: dict) -> None:
    caches = {c["cache"]: {(r["method"], r["k"]): r["mean"] for r in c["per_k"]}
              for c in trend["moderate_caches"]}
    assert sorted(caches) == sorted(trend["scopes"][MODERATE]["caches"])
    for cache, means in caches.items():
        for method in ("split", "mondrian"):
            assert means[(method, 10)] > 0, (cache, method)


def test_brightness_s1_source_mondrian_is_within_noise_of_the_reference(trend: dict) -> None:
    """Noise is the seed-mean band of one cache: band_z reference sd over root seeds."""
    means = next({(r["method"], r["k"]): r["mean"] for r in c["per_k"]}
                 for c in trend["moderate_caches"] if c["cache"] == "brightness_s1")
    ref, z, seeds = _reference(trend), trend["config"]["band_z"], trend["config"]["n_seeds"]
    for k in (10, 100):
        half = z * trend["reference"][str(k)]["sd"] / math.sqrt(seeds)
        assert abs(means[("mondrian", k)] - ref[k]) <= half, k


def test_split_and_ppi_are_below_on_every_cache(pool: dict[str, str], trend: dict) -> None:
    n = int(pool["ksweep_all.n_caches"])
    assert n == trend["config"]["n_files"] == 80
    for method in ("split", "ppi"):
        assert int(pool[f"ksweep_all.K100.{method}.n_below"]) == n, method


def test_source_mondrian_is_never_above_the_band(pool: dict[str, str]) -> None:
    """Within the band on some caches and below it on the others."""
    n = int(pool["ksweep_all.n_caches"])
    assert int(pool["ksweep_all.K100.mondrian.n_above"]) == 0
    assert (int(pool["ksweep_all.K100.mondrian.n_within"])
            + int(pool["ksweep_all.K100.mondrian.n_below"])) == n


def test_oracle_stays_within_one_cache_noise_of_the_reference(trend: dict) -> None:
    z, seeds = trend["config"]["band_z"], trend["config"]["n_seeds"]
    for scope in BACKBONES:
        rows = _rows(trend, scope)
        for k in K_VALUES:
            half = z * trend["reference"][str(k)]["sd"] / math.sqrt(seeds)
            assert abs(rows[("oracle", k)]["diff"]) <= half, (scope, k)


def test_more_than_half_of_source_mondrian_vitb_cells_read_zero_at_every_k(trend: dict) -> None:
    rows = _rows(trend, "ViT-B/32")
    for k in K_VALUES:
        assert rows[("mondrian", k)]["zero_share"] > 0.5, k


def test_of_the_label_free_methods_only_split_and_ppi_fall_further(trend: dict) -> None:
    """The shortfall below the reference grows from K=10 to K=100 for split and PPI and does not
    grow for Source-Mondrian and weighted, on both backbones."""
    for scope in BACKBONES:
        rows = _rows(trend, scope)
        growth = {m: rows[(m, 10)]["diff"] - rows[(m, 100)]["diff"]
                  for m in ("split", "ppi", "mondrian", "weighted")}
        assert growth["split"] > 0.03 and growth["ppi"] > 0.03, (scope, growth)
        assert growth["mondrian"] <= 0 and growth["weighted"] <= 0, (scope, growth)


@functools.lru_cache(maxsize=1)
def _sweep_records() -> dict[str, list[dict]]:
    return {path.stem.removeprefix("controlled_k_sweep_"):
            json.loads(path.read_text(encoding="utf-8"))["records"]
            for path in sorted(_ANALYSIS.glob("controlled_k_sweep_*.json"))}


def _per_cache_means(method: str, k: int) -> dict[str, float]:
    return {cache: mean(r["worst_class"] for r in records if r["method"] == method and r["k"] == k)
            for cache, records in _sweep_records().items()}


def test_named_floored_caches_are_at_or_near_zero_at_k10() -> None:
    near_zero = 0.10
    at_k10 = {m: _per_cache_means(m, 10) for m in ("split", "mondrian", "ppi")}
    assert all(at_k10[m]["gaussian_noise_s3"] == 0 for m in at_k10)
    assert all(at_k10[m]["fog_s3"] <= near_zero for m in at_k10)
    assert all(at_k10[m]["jpeg_compression_s1"] <= near_zero for m in ("split", "mondrian"))


def test_gaussian_noise_s3_oracle_is_the_lowest_of_all_caches(pool: dict[str, str]) -> None:
    oracle = _per_cache_means("oracle", 100)
    assert len(oracle) == int(pool["ksweep_all.n_caches"])
    assert min(oracle, key=oracle.get) == "gaussian_noise_s3"
    assert f"{oracle['gaussian_noise_s3']:.3f}" == pool["sweep.floored.gaussian_noise_s3.K100.mean"]


def test_the_five_caches_are_a_hand_picked_part_of_the_qualifying_caches(
        pool: dict[str, str], trend: dict) -> None:
    """Split and Source-Mondrian keep a mean worst-class coverage above zero at K=10 on more
    caches than the five, and brightness severity two is one of the others."""
    split, mondrian = _per_cache_means("split", 10), _per_cache_means("mondrian", 10)
    vitb = [cache for cache in split if not cache.startswith("vitL_")]
    qualifying = {cache for cache in vitb if split[cache] > 0 and mondrian[cache] > 0}
    picked = set(trend["scopes"][MODERATE]["caches"])
    assert len(vitb) == int(pool["derived.selection.n_vitb_caches"])
    assert len(qualifying) == int(pool["derived.selection.n_qualifying"])
    assert len(picked) == int(pool["derived.selection.n_picked"]) == 5
    assert picked < qualifying
    assert "brightness_s2" in qualifying - picked


def test_bonferroni_over_the_twenty_comparisons_loses_one_statement(trend: dict) -> None:
    """The paragraph states intervals of five methods at four K."""
    ref = _reference(trend)
    caches = trend["scopes"][MODERATE]["caches"]
    records = _sweep_records()
    seeds = sorted({r["subset_seed"] for r in records[caches[0]]})
    methods = ("split", "mondrian", "weighted", "ppi", "acp")
    family = len(methods) * len(K_VALUES)
    assert family == 20

    def position(method: str, k: int, level: float) -> str:
        per_seed = [mean(r["worst_class"] for cache in caches for r in records[cache]
                         if r["method"] == method and r["k"] == k and r["subset_seed"] == seed)
                    for seed in seeds]
        half = stats.t.ppf(1 - level / 2, len(seeds) - 1) * stdev(per_seed) / math.sqrt(len(seeds))
        if mean(per_seed) + half < ref[k]:
            return "below"
        return "above" if mean(per_seed) - half > ref[k] else "contains"

    stated = {(m, k): "below" for m in ("split", "mondrian") for k in K_VALUES}
    stated |= {(m, k): "below" for m in ("weighted", "ppi") for k in (25, 50, 100)}
    stated |= {("acp", k): "above" for k in K_VALUES}
    for (method, k), claim in stated.items():
        assert position(method, k, 0.05) == claim, (method, k)
    lost = {key for key, claim in stated.items() if position(*key, 0.05 / family) != claim}
    assert lost == {("split", 10)}
    assert all(position(method, 100, 0.05 / family) == stated[(method, 100)] for method in methods)


def test_the_oracle_band_count_is_about_the_chance_rate(pool: dict[str, str]) -> None:
    n = int(pool["ksweep_all.n_caches"])
    by_chance = n * (stats.norm.cdf(2) - stats.norm.cdf(-2))
    assert abs(int(pool["ksweep_all.K100.oracle.n_within"]) - by_chance) <= 2


def test_the_gaussian_noise_excursion_is_of_the_size_chance_produces(pool: dict[str, str]) -> None:
    """Among as many means as there are caches, at least one this far out is expected."""
    gap = _num(pool, "sweep.floored.gaussian_noise_s3.K100.gap_in_seed_mean_se.magnitude")
    assert int(pool["ksweep_all.n_caches"]) * 2 * stats.norm.sf(gap) >= 1


NESTED_FAMILY_METHODS = ("split", "mondrian", "weighted", "ppi", "acp")
WIDENING_METHODS = ("split", "ppi", "weighted", "mondrian")


def _decomposition(nested: dict, scope: str) -> dict[str, dict]:
    return {r["method"]: r for r in nested["scopes"][scope]["decomposition"]}


def _family_readings(cr: ModuleType, nested: dict) -> dict[tuple[str, str, str], str]:
    family = nested["config"]["family"]
    return {(scope, method, term): cr.term_reading(_decomposition(nested, scope)[method], term)
            for scope in family["scopes"] for method in family["methods"]
            for term in family["terms"]}


def test_the_nested_run_is_the_one_the_appendix_describes(pool: dict[str, str], trend: dict,
                                                          nested: dict) -> None:
    config = nested["config"]
    assert config["design"] == "nested"
    assert (config["n_files"], config["n_seeds"]) == (80, 30)
    assert pool["nested.n_files"] == pool["ksweep_all.n_caches"]
    assert config["acp_tie_breaks"] == ["random"]
    family = config["family"]
    assert tuple(sorted(family["methods"])) == tuple(sorted(NESTED_FAMILY_METHODS))
    assert family["scopes"] == list(BACKBONES) and tuple(family["k_pair"]) == (10, 100)
    assert family["size"] == len(NESTED_FAMILY_METHODS) * 2 * 2 == 20
    assert family["bound_share"] == 0.5
    assert {nested["no_shift"][b]["n_views"] for b in BACKBONES} == {5}
    assert nested["scopes"][MODERATE]["selected_on_outcomes"] is True
    assert [nested["scopes"][b]["n_caches"] for b in BACKBONES] == [40, 40]
    assert nested["scopes"][MODERATE]["caches"] == trend["scopes"][MODERATE]["caches"]
    assert nested["reference"] == trend["reference"]


def test_the_two_terms_sum_to_the_change_in_the_difference_from_the_reference(nested: dict) -> None:
    """For every method on every scope, and against the per-K rows."""
    for scope in (*BACKBONES, MODERATE):
        per_k = _rows(nested, scope)
        for method, row in _decomposition(nested, scope).items():
            total = row["same_class"]["mean"] + row["selection"]["mean"]
            assert total == pytest.approx(row["total"], abs=1e-9), (scope, method)
            change = per_k[(method, 100)]["diff"] - per_k[(method, 10)]["diff"]
            assert total == pytest.approx(change, abs=1e-9), (scope, method)


def test_the_nested_oracle_shows_no_term(nested: dict) -> None:
    """Both terms have expectation zero under per-class validity."""
    for scope in (*BACKBONES, MODERATE):
        row = _decomposition(nested, scope)["oracle"]
        for term in ("same_class", "selection"):
            assert row[term]["ci_lo"] <= 0 <= row[term]["ci_hi"], (scope, term)


def test_source_mondrian_widens_by_selection_on_the_five_caches(nested: dict) -> None:
    """The same classes are covered no worse, the added classes carry the change, and neither
    term moves without shift."""
    row = _decomposition(nested, MODERATE)["mondrian"]
    same, selection = row["same_class"], row["selection"]
    assert same["ci_lo"] <= 0 <= same["ci_hi"]
    assert selection["ci_hi"] < 0
    assert abs(selection["mean"]) > 5 * abs(same["mean"])
    for term in (same, selection):
        assert term["no_shift"]["ci_lo"] <= 0 <= term["no_shift"]["ci_hi"]
        assert term["bound_share"] < 0.05
    assert selection["minus_no_shift"]["ci_hi"] < 0
    assert same["minus_no_shift"]["ci_lo"] <= 0 <= same["minus_no_shift"]["ci_hi"]


def test_the_contrast_caches_carry_source_mondrian_selection_term(pool: dict[str, str],
                                                                  nested: dict) -> None:
    """Almost all of the selection term of Source-Mondrian comes from the three contrast caches,
    and the five printed values are the means of the five caches."""
    caches = nested["scopes"][MODERATE]["caches"]
    means = dict(zip(caches, _decomposition(nested, MODERATE)["mondrian"]["selection"]["caches_means"],
                     strict=True))
    contrast = [v for c, v in means.items() if c.startswith("contrast_")]
    brightness = [v for c, v in means.items() if c.startswith("brightness_")]
    assert (len(contrast), len(brightness)) == (3, 2)
    assert all(v < -0.15 for v in contrast) and all(abs(v) < 0.03 for v in brightness)
    assert sum(contrast) / sum(means.values()) > 0.95
    for cache, value in means.items():
        printed = pool[f"nested.cache.{cache}.mondrian.decomp.selection.mean"]
        assert float(printed) == pytest.approx(value, abs=5e-4), cache


def test_split_weighted_and_ppi_fall_at_least_as_far_without_shift(nested: dict) -> None:
    """On the five caches their difference from the reference falls from K=10 to K=100 by no more
    than it falls on the no-shift views of the same backbone."""
    shifted = _rows(nested, MODERATE)
    plain = {(r["method"], r["k"]): r for r in nested["no_shift"]["ViT-B/32"]["per_k"]}
    for method in ("split", "weighted", "ppi"):
        under_shift = shifted[(method, 100)]["diff"] - shifted[(method, 10)]["diff"]
        without = plain[(method, 100)]["diff"] - plain[(method, 10)]["diff"]
        assert without <= under_shift < 0, (method, under_shift, without)
        row = _decomposition(nested, MODERATE)[method]
        no_shift_sum = sum(row[t]["no_shift"]["mean"] for t in ("same_class", "selection"))
        assert no_shift_sum <= row["total"] < 0, method
    for method in ("split", "weighted"):
        assert shifted[(method, 100)]["mean"] < 0.15, method


def test_source_mondrian_terms_cannot_be_read_over_all_caches(cr: ModuleType, pool: dict[str, str],
                                                              nested: dict) -> None:
    """Most of its cells sit at zero, three of its four family terms are bound, and the fourth is
    above zero through its pinned cells only."""
    readings = _family_readings(cr, nested)
    own = {key: r for key, r in readings.items() if key[1] == "mondrian"}
    assert sorted(own.values()) == ["bound", "bound", "bound", "withheld"]
    assert pool["derived.nested.family.mondrian.n_bound"] == "3"
    assert pool["derived.nested.family.mondrian.n_withheld"] == "1"
    (scope, _, term), = [key for key, r in own.items() if r == "withheld"]
    stat = _decomposition(nested, scope)["mondrian"][term]
    assert term == "selection" and stat["excludes_zero"] == 1
    assert stat["free_mean"] < 0 < stat["pinned_part"]
    for scope in BACKBONES:
        for k in (10, 100):
            assert _rows(nested, scope)[("mondrian", k)]["zero_share"] > 0.5, (scope, k)


_B, _L = BACKBONES
EXPECTED_READINGS = {
    (_B, "split", "same_class"): "bound", (_B, "split", "selection"): "negative",
    (_B, "mondrian", "same_class"): "bound", (_B, "mondrian", "selection"): "bound",
    (_B, "weighted", "same_class"): "positive", (_B, "weighted", "selection"): "negative",
    (_B, "ppi", "same_class"): "negative", (_B, "ppi", "selection"): "bound",
    (_B, "acp", "same_class"): "bound", (_B, "acp", "selection"): "positive",
    (_L, "split", "same_class"): "bound", (_L, "split", "selection"): "negative",
    (_L, "mondrian", "same_class"): "bound", (_L, "mondrian", "selection"): "withheld",
    (_L, "weighted", "same_class"): "positive", (_L, "weighted", "selection"): "negative",
    (_L, "ppi", "same_class"): "negative", (_L, "ppi", "selection"): "bound",
    (_L, "acp", "same_class"): "bound", (_L, "acp", "selection"): "positive",
}


def test_the_reading_of_every_family_term_is_the_one_written_down(cr: ModuleType, nested: dict) -> None:
    assert _family_readings(cr, nested) == EXPECTED_READINGS
    for (scope, method, term), reading in EXPECTED_READINGS.items():
        share = _decomposition(nested, scope)[method][term]["bound_share"]
        assert (share >= 0.5) == (reading == "bound"), (scope, method, term, share)


def test_the_family_readings_are_the_ones_the_appendix_names(cr: ModuleType, pool: dict[str, str],
                                                             nested: dict) -> None:
    readings = _family_readings(cr, nested)
    counts = Counter(readings.values())
    assert counts == Counter(EXPECTED_READINGS.values())
    assert counts["bound"] == 9 and counts["withheld"] == 1
    assert sum(counts.values()) == int(pool["nested.family.size"]) == 20
    for reading in ("bound", "contains", "withheld", "negative", "positive"):
        assert pool[f"derived.nested.family.n_{reading}"] == str(counts[reading]), reading
    assert counts["outside"] == 0 and counts["contains"] == 0
    negative = {(method, term) for (_, method, term), r in readings.items() if r == "negative"}
    positive = {(method, term) for (_, method, term), r in readings.items() if r == "positive"}
    assert negative == {("split", "selection"), ("weighted", "selection"), ("ppi", "same_class")}
    assert positive == {("acp", "selection"), ("weighted", "same_class")}
    assert counts["negative"] == 2 * len(negative) and counts["positive"] == 2 * len(positive)


def test_every_negative_family_term_is_closer_to_zero_than_without_shift(
        cr: ModuleType, pool: dict[str, str], nested: dict) -> None:
    """The paired interval of the term under shift minus the term without shift lies above zero
    for each of them, so none goes beyond its no-shift value."""
    readings = _family_readings(cr, nested)
    negative = [key for key, r in readings.items() if r == "negative"]
    for scope, method, term in negative:
        stat = _decomposition(nested, scope)[method][term]
        assert stat["minus_no_shift"]["ci_lo"] > 0, (scope, method, term)
        assert stat["no_shift"]["mean"] < stat["mean"] < 0, (scope, method, term)
    assert pool["derived.nested.family.n_negative_short_of_no_shift"] == str(len(negative))
    assert pool["derived.nested.family.n_negative_beyond_no_shift"] == "0"


def test_the_positive_family_terms_belong_to_methods_with_large_sets(nested: dict) -> None:
    """At K=100 the sets of ACP and weighted hold more than half of the label space on both
    backbones."""
    for scope in BACKBONES:
        rows = _rows(nested, scope)
        assert rows[("acp", 100)]["set_over_k"] > 0.9, scope
        assert rows[("weighted", 100)]["set_over_k"] > 0.5, scope


def test_a_bound_term_has_no_printable_value(cr: ModuleType, pool: dict[str, str],
                                             nested: dict) -> None:
    """The prose cannot quote a term the reading rules do not interpret: such a term, inside the
    family or outside it, has its share of bound cells in the pool and no other key."""
    n_bound = {True: 0, False: 0}
    for scope, slug in cr.ALL_SCOPE_SLUG.items():
        for method, row in _decomposition(nested, scope).items():
            for term in ("same_class", "selection"):
                base = f"nested.{slug}.{method}.decomp.{term}."
                keys = {k.removeprefix(base) for k in pool if k.startswith(base)}
                bound = row[term]["bound_share"] >= 0.5
                assert bound == row[term]["marked"]
                if bound:
                    assert keys == {"bound_share"}, (scope, method, term, sorted(keys))
                    n_bound[row["in_family"]] += 1
                else:
                    assert {"bound_share", "mean", "ci_lo", "ci_hi", "free_mean",
                            "n_caches_negative", "no_shift.mean", "minus_no_shift.ci_lo",
                            "minus_no_shift.ci_hi"} <= keys, (scope, method, term)
    assert n_bound[True] == 9 and n_bound[False] >= 1


def test_the_nested_run_reads_lower_at_k10_and_agrees_elsewhere(pool: dict[str, str], trend: dict,
                                                               nested: dict) -> None:
    """At K=10 the four differences of the nested run go from split's to Source-Mondrian's, each
    30-seed interval is below the reference, and each mean is inside its five-seed interval and
    under the five-seed mean, split's and PPI's at the lower edge."""
    old, new = _rows(trend, MODERATE), _rows(nested, MODERATE)
    ref = _reference(nested)
    diffs = {m: new[(m, 10)]["diff"] for m in WIDENING_METHODS}
    assert min(diffs, key=diffs.get) == "split" and max(diffs, key=diffs.get) == "mondrian"
    for method in WIDENING_METHODS:
        assert new[(method, 10)]["ci_hi"] < ref[10], method
        assert old[(method, 10)]["ci_lo"] <= new[(method, 10)]["mean"] < old[(method, 10)]["mean"]
    for method in ("split", "ppi"):
        assert new[(method, 10)]["mean"] - old[(method, 10)]["ci_lo"] < 0.005, method
    others = ("oracle", "mondrian", "split", "weighted", "ppi", "acp")
    for method in others:
        for k in K_VALUES:
            assert old[(method, k)]["ci_lo"] <= new[(method, k)]["mean"] <= old[(method, k)]["ci_hi"], \
                (method, k)
    assert pool["derived.nested.replication.n_methods"] == str(len(others))
    assert pool["derived.nested.replication.K10.n_inside"] == str(len(others))
    assert pool["derived.nested.replication.n_cells_inside"] == \
        pool["derived.nested.replication.n_cells"] == str(len(others) * len(K_VALUES))
    bound = _num(pool, "derived.nested.replication.K100.max_abs_diff")
    largest = max(abs(new[(method, 100)]["mean"] - old[(method, 100)]["mean"]) for method in others)
    assert largest <= bound < largest + 1e-3


def test_the_five_seed_reach_at_k10_is_not_confirmed_by_thirty_seeds(trend: dict, nested: dict) -> None:
    """The five-seed K=10 intervals of weighted and PPI reach the reference, the 30-seed
    intervals of the nested run lie below it."""
    old, new = _rows(trend, MODERATE), _rows(nested, MODERATE)
    ref = _reference(trend)
    for method in ("weighted", "ppi"):
        assert old[(method, 10)]["ci_lo"] <= ref[10] <= old[(method, 10)]["ci_hi"], method
        assert new[(method, 10)]["ci_hi"] < ref[10] - 0.2, method


def test_the_widening_is_smaller_in_the_nested_run_and_remains(trend: dict, nested: dict) -> None:
    old, new = _rows(trend, MODERATE), _rows(nested, MODERATE)
    for method in WIDENING_METHODS:
        widening_new = new[(method, 100)]["diff"] - new[(method, 10)]["diff"]
        widening_old = old[(method, 100)]["diff"] - old[(method, 10)]["diff"]
        assert widening_old < widening_new < 0, (method, widening_old, widening_new)


def test_the_negative_family_terms_by_corruption_family_and_by_bound_share(
        pool: dict[str, str], nested: dict) -> None:
    """PPI's same-class term is negative in all eight corruption families on both backbones,
    split's selection term in two on both, weighted's in three and five; and the six negative
    terms have the printed range of bound shares."""
    vitb, vitl = (_decomposition(nested, scope) for scope in BACKBONES)
    assert [d["ppi"]["same_class"]["n_families_negative"] for d in (vitb, vitl)] == [8, 8]
    assert [d["split"]["selection"]["n_families_negative"] for d in (vitb, vitl)] == [2, 2]
    assert [d["weighted"]["selection"]["n_families_negative"] for d in (vitb, vitl)] == [3, 5]
    assert pool["nested.vitb.ppi.decomp.same_class.n_families_negative"] == "8"
    assert pool["nested.vitb.split.decomp.selection.n_families_negative"] == "2"
    assert pool["nested.vitb.weighted.decomp.selection.n_families_negative"] == "3"
    assert pool["nested.vitl.weighted.decomp.selection.n_families_negative"] == "5"
    shares = [_decomposition(nested, scope)[method][term]["bound_share"]
              for (scope, method, term), reading in EXPECTED_READINGS.items() if reading == "negative"]
    assert len(shares) == 6
    assert pool["derived.nested.family.negative.min_bound_share"] == f"{min(shares):.2f}"
    assert pool["derived.nested.family.negative.max_bound_share"] == f"{max(shares):.2f}"
    assert 0.2 < min(shares) and max(shares) < 0.5


def test_the_caches_of_a_backbone_share_source_data_images_and_class_orders() -> None:
    """Read from the raw files of the nested run: one source calibration set, one test set size
    and one class order per seed for every cache of a backbone."""
    import gzip

    for prefix in ("", "vitL_"):
        seen = set()
        for path in sorted((_ROOT / "outputs" / "nested_ksweep").glob("nested_k_sweep_*.json.gz")):
            name = path.name.removeprefix("nested_k_sweep_")
            if name.startswith("vitL_") != bool(prefix):
                continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            provenance = payload["provenance"]
            seen.add((provenance["source_cal_sha256"], provenance["n_target_test"],
                      json.dumps(payload["class_orders"])))
            if len(seen) > 1:
                break
        assert len(seen) == 1, prefix


def _compact(x: float) -> str:
    text = f"{x:+.3f}"
    if text in ("+0.000", "-0.000"):
        return ".000"
    return text[0] + text[2:] if text[1:3] == "0." else text


def test_the_nested_table_prints_the_terms_of_the_five_caches(cr: ModuleType, nested: dict) -> None:
    """The table prints each nested term's mean under shift and without shift."""
    body = cr.tab_nested(_ROOT).splitlines()
    methods = ("oracle", "mondrian", "split", "weighted", "ppi", "acp")
    assert len(body) == len(methods)
    rows = _decomposition(nested, MODERATE)
    for line, method in zip(body, methods):
        assert line.endswith(" \\\\")
        cells = line.removesuffix(" \\\\").split(" & ")
        expected = [cr.METHOD_NAME[method]]
        for term in ("same_class", "selection"):
            stat = rows[method][term]
            expected.append(f"${_compact(stat['mean'])}$")
            expected.append(f"${_compact(stat['no_shift']['mean'])}$")
        assert cells == expected, method


def test_the_compact_signed_format_keeps_sign_and_magnitude(cr: ModuleType) -> None:
    for value, text in ((-0.1169, "-.117"), (0.0101, "+.010"), (0.0001, ".000"), (-0.0001, ".000"),
                        (1.0, "+1.000"), (-1.25, "-1.250"), (10.5, "+10.500"), (0.5, "+.500")):
        assert cr.sdot3(value) == text == _compact(value), value


def test_oracle_cross_dataset_decline_is_that_of_the_reference(pool: dict[str, str]) -> None:
    reference = _num(pool, "derived.crossdataset.reference_decline")
    for backbone in ("vitb", "vitl"):
        assert abs(_num(pool, f"derived.vitl.oracle.{backbone}.magnitude") - reference) <= 0.01


def test_ppi_declines_more_and_acp_split_and_mondrian_less_than_the_reference(
        pool: dict[str, str]) -> None:
    """On ViT-B/32 PPI declines by more than the reference and ACP, split and Source-Mondrian by
    less; on ViT-L/14 PPI by more and ACP by less."""
    reference = _num(pool, "derived.crossdataset.reference_decline")
    for method in ("acp", "ppi", "split", "mondrian"):
        assert _num(pool, f"derived.delta.{method}.informative") < 0, method
    assert _num(pool, "derived.delta.ppi.informative.magnitude") > reference
    for method in ("acp", "split", "mondrian"):
        assert _num(pool, f"derived.delta.{method}.informative.magnitude") < reference, method
    assert _num(pool, "derived.vitl.ppi.vitl") < 0 and _num(pool, "derived.vitl.acp.vitl") < 0
    assert _num(pool, "derived.vitl.ppi.vitl.magnitude") > reference
    assert _num(pool, "derived.vitl.acp.vitl.magnitude") < reference


@pytest.mark.parametrize("method", ["acp", "ppi", "split", "mondrian"])
def test_the_cross_dataset_declines_are_those_of_the_row_files(pool: dict[str, str], method: str) -> None:
    """Without the generator: the informative pairs and their mean change, recomputed from the
    per-row outputs of both backbones, give the printed count and the printed cells, and the same
    order against the reference decline."""
    reference = _num(pool, "derived.crossdataset.reference_decline")
    vitb, vitl = _informative_pairs(method, ""), _informative_pairs(method, "vitL_")
    assert str(len(vitb)) == pool[f"derived.delta.{method}.n_info"]
    assert f"{_mean_change(vitb):+.3f}" == pool[f"derived.delta.{method}.informative"]
    assert str(len(vitl)) == pool[f"derived.vitl.{method}.n_info"]
    assert f"{_mean_change(vitl):+.3f}" == pool[f"derived.vitl.{method}.vitl"]
    assert (-_mean_change(vitb) > reference) == (method == "ppi")
    if method != "split":
        assert (-_mean_change(vitl) > reference) == (method == "ppi")


def test_weighted_informative_mean_averages_two_opposite_groups(pool: dict[str, str]) -> None:
    """Pairs with near-full CIFAR-100-C sets that the guard demotes, and the other pairs, on
    which weighted declines by more than the reference, on both backbones."""
    reference = _num(pool, "derived.crossdataset.reference_decline")
    for slug, n_info in (("vitb", "derived.delta.weighted.n_info"),
                         ("vitl", "derived.vitl.weighted.n_info")):
        base = f"derived.weighted_groups.{slug}"
        n_full, n_rest = int(pool[f"{base}.n_full"]), int(pool[f"{base}.n_rest"])
        assert n_full + n_rest == int(pool[n_info]), slug
        assert int(pool[f"{base}.n_full_demoted"]) == n_full, slug
        assert 0.95 <= _num(pool, f"{base}.full_min_worst") < 0.99, slug
        assert _num(pool, f"{base}.rest_mean") < 0, slug
        assert _num(pool, f"{base}.rest_mean.magnitude") > reference, slug
    for key in ("derived.delta.weighted.informative", "derived.vitl.weighted.vitl"):
        assert abs(_num(pool, key)) < 0.01, key


def test_regime_b_is_the_largest_category(pool: dict[str, str]) -> None:
    counts = {c: int(pool[f"thresholds.default_counts.{c}"])
              for c in ("recovered", "regime_a", "regime_b", "other")}
    assert sum(counts.values()) == int(pool["thresholds.default_counts.non_demoted"])
    assert counts["regime_b"] > max(v for c, v in counts.items() if c != "regime_b")


def test_every_oracle_and_ppi_aps_twin_keeps_its_regime(pool: dict[str, str]) -> None:
    assert pool["aps.oracle_ppi.n_same_regime"] == pool["aps.oracle_ppi.n_rows_with_twin"]


def test_the_fixed_threshold_is_strict_on_cifar100c_and_lenient_on_cifar10c(
        pool: dict[str, str]) -> None:
    assert (_num(pool, "floor.cifar100c.K100.floor_mean") < 0.80
            < _num(pool, "floor.cifar10c.K10.floor_mean"))


def test_regime_a_holds_every_cifar100c_oracle_row_and_nearly_all_sit_on_the_reference(
        pool: dict[str, str]) -> None:
    n_rows = int(pool["derived.oracle.cifar100c.n_rows"])
    assert n_rows == int(pool["regime.floor_rule.regime_a.oracle_cifar100c.n_rows"])
    assert 0.9 * n_rows <= int(pool["derived.oracle.cifar100c.n_within_two_se"]) <= n_rows


def test_regime_b_does_not_survive_the_aps_score(pool: dict[str, str]) -> None:
    """Under APS most regime-B rows are demoted for near-full sets and few stay in regime B."""
    n = int(pool["aps.regime_b.n_rows_with_twin"])
    assert n == int(pool["thresholds.default_counts.regime_b"])
    assert int(pool["aps.regime_b.n_twins_demoted"]) > n / 2
    assert int(pool["aps.regime_b.n_keep_regime_b"]) < n / 10


def test_the_threshold_sweep_varies_one_family_at_a_time(pool: dict[str, str]) -> None:
    assert (int(pool["thresholds.n_regime_combinations"])
            + int(pool["thresholds.n_guard_combinations"])) == int(pool["thresholds.n_combinations"])


def test_oracle_label_intervals_contain_the_reference(pool: dict[str, str]) -> None:
    for m in (10, 25, 50):
        ref = _num(pool, f"floor.label_sweep.K100.ncal{m}.floor_mean")
        assert _num(pool, f"derived.label.oracle.m{m}.ci_lo") <= ref <= \
            _num(pool, f"derived.label.oracle.m{m}.ci_hi"), m


def test_ppi_is_below_the_oracle_at_10_labels_and_not_separated_at_50(pool: dict[str, str]) -> None:
    base = "derived.label.ppi_minus_oracle"
    assert _num(pool, f"{base}.m10.diff_ci_hi") < 0
    assert _num(pool, f"{base}.m50.diff_ci_lo") <= 0 <= _num(pool, f"{base}.m50.diff_ci_hi")
    assert _num(pool, "sweep.label.ppi.m0.mean") == 0


def test_acp_is_above_the_reference_at_every_budget_with_near_full_sets(
        pool: dict[str, str]) -> None:
    """At 10, 25 and 50 labels ACP's ten-seed t interval lies above the reference and its sets
    hold at least 0.90 of the 100 classes, the first condition of the efficiency guard."""
    budgets = (10, 25, 50)
    for m in budgets:
        reference = _num(pool, f"floor.label_sweep.K100.ncal{m}.floor_mean")
        assert _num(pool, f"derived.label.acp.m{m}.ci_lo") > reference, m
        assert _num(pool, f"label_sweep.set_size.acp.m{m}") >= 0.9 * 100, m
    means = [_num(pool, f"sweep.label.acp.m{m}.mean") for m in budgets]
    assert min(means) == means[-1] and max(means) == means[0]
    label = json.loads((_ANALYSIS / "label_complexity_contrast_s3.json").read_text(encoding="utf-8"))
    exact = {m: mean(r["worst_class"] for r in label["per_seed"] if r["method"] == "acp" and r["m"] == m)
             for m in budgets}
    assert all(len([r for r in label["per_seed"] if r["method"] == "acp" and r["m"] == m]) == 10
               for m in budgets)
    assert exact[50] <= exact[25] <= exact[10]
    for m in budgets:
        assert f"{exact[m]:.3f}" == pool[f"sweep.label.acp.m{m}.mean"], m


def test_acp_is_above_the_band_on_every_cache_with_large_sets(pool: dict[str, str],
                                                             trend: dict) -> None:
    """At K=100 ACP's five-seed mean is more than two reference standard errors above the
    reference on all 80 combinations, and its mean set holds the printed range of the label
    space, never less than 0.70 of it."""
    n = int(pool["ksweep_all.n_caches"])
    assert int(pool["ksweep_all.K100.acp.n_above"]) == n
    assert int(pool["ksweep_all.K100.acp.n_below"]) == 0
    ref, z, seeds = _reference(trend), trend["config"]["band_z"], trend["config"]["n_seeds"]
    half = z * trend["reference"]["100"]["sd"] / math.sqrt(seeds)
    means = _per_cache_means("acp", 100)
    assert len(means) == n
    assert all(value > ref[100] + half for value in means.values())
    shares = {cache: mean(r["set_over_k"] for r in records if r["method"] == "acp" and r["k"] == 100)
              for cache, records in _sweep_records().items()}
    assert f"{min(shares.values()):.3f}" == pool["ksweep_all.K100.acp.min_set_over_k_above"]
    assert f"{max(shares.values()):.3f}" == pool["ksweep_all.K100.acp.max_set_over_k_above"]
    assert min(shares.values()) >= 0.70
    assert _num(pool, "ksweep_all.K100.weighted.min_set_over_k_above") >= 0.70


def test_every_demoted_row_would_be_labelled_recovered(pool: dict[str, str]) -> None:
    """Before demotion every demoted row meets the recovered threshold."""
    assert pool["thresholds.default_counts.demoted"] == pool["thresholds.default_counts.demoted_recovered"]
    assert pool["thresholds.default_counts.demoted_regime_a"] == "0"


def test_imagenet_acp_reads_above_zero_on_every_cell_and_one_cell_is_demoted(pool: dict[str, str]) -> None:
    """On ImageNet-1k ACP's seed-mean worst-class coverage is above zero on all ten cells, its
    sets hold the printed range of the classes, and the guard demotes one cell."""
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((_ROOT / "outputs").glob("imagenet1k_*_acp.json"))]
    assert len(rows) == int(pool["derived.imagenet.acp.n_cells"]) == 10
    assert all(row["worst_class"] > 0 for row in rows)
    assert pool["derived.imagenet.acp.n_zero_worst"] == "0"
    assert f"{mean(row['worst_class'] for row in rows):.3f}" == pool["derived.imagenet.acp.worst_class"]
    shares = [row["mean_set_size"] / row["n_classes"] for row in rows]
    assert f"{min(shares):.2f}" == pool["derived.imagenet.acp.min_set_over_k"]
    assert f"{max(shares):.2f}" == pool["derived.imagenet.acp.max_set_over_k"]
    demoted = [share >= 0.90 or (share >= 0.60 and row["marginal"] >= 0.95)
               for share, row in zip(shares, rows, strict=True)]
    assert sum(demoted) == int(pool["derived.imagenet.acp.n_demoted"]) == 1
    weighted = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((_ROOT / "outputs").glob("imagenet1k_*_weighted.json"))]
    assert len(weighted) == 10
    assert sum(row["worst_class"] == 0 for row in weighted) == \
        int(pool["derived.imagenet.weighted.n_zero_worst"]) > 0
    assert sum(_guard_demotes(row) for row in weighted) == int(pool["derived.imagenet.weighted.n_demoted"]) == 1


def test_every_printed_acp_value_comes_from_a_run_under_the_random_tie_order(cr: ModuleType) -> None:
    """Every ACP row of the board, each of the 80 controlled sweeps and the label sweep record
    the random tie order."""
    registry = cr.registry_module
    board = registry.rerun_board(_ROOT, registry.analyze_module(_ROOT))
    acp_rows = {row.dataset: row for row in board if row.method == "acp"}
    assert len(acp_rows) == 175
    for dataset, shown in acp_rows.items():
        row = _row_file(dataset, "acp")
        assert row["tie_break"] == "random", dataset
        assert shown.marginal == float(f"{row['marginal']:.3f}"), dataset
        assert shown.worst_class == float(f"{row['worst_class']:.3f}"), dataset
        assert shown.set_over_k == float(f"{_set_share(row):.2f}"), dataset
    assert "acp" not in cr.NO_SHIFT_METHODS
    sweeps = sorted(_ANALYSIS.glob("controlled_k_sweep_*.json"))
    assert len(sweeps) == 80
    for path in sweeps:
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        assert config["acp_tie_break"] == "random", path.name
    label = json.loads((_ANALYSIS / "label_complexity_contrast_s3.json").read_text(encoding="utf-8"))
    assert label["acp_tie_break"] == "random"


def test_ntu_methods_leave_the_same_action_uncovered_on_every_seed(pool: dict[str, str]) -> None:
    zero_classes = set()
    for method in ("split", "mondrian", "weighted"):
        row = json.loads((_ROOT / "outputs" / f"ntu2nucla_{method}.json").read_text(encoding="utf-8"))
        per_seed = row["per_class_coverage_per_seed"]
        assert len(per_seed) == row["seeds"] == 10
        never = {c for c in range(row["n_classes"]) if all(seed[c] == 0 for seed in per_seed)}
        assert len(never) == 1, (method, never)
        zero_classes |= never
    assert len(zero_classes) == 1
    assert pool["derived.ntu2nucla.labelfree.worst_class"] == "0.000"
