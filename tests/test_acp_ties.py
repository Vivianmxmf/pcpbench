"""ACP's audited probabilities give every class but the predicted one the same mass, so the APS
order of those classes is an exact tie."""

from __future__ import annotations

import numpy as np
import pytest

from pcpbench.calibrators.acp import ACPCalibrator, _audit_probs
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.metrics.coverage import per_class_coverage
from pcpbench.scores.softmax_scores import aps_scores, predict_sets_from_softmax


def _tied_rows(n: int, k: int, r: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = np.zeros((n, k))
    logits[np.arange(n), rng.integers(0, k, n)] = 5.0
    return _audit_probs(logits, np.full(n, r))


def _inclusion_of_tied_classes(m: np.ndarray, threshold: float, tie_rng) -> np.ndarray:
    n, k = m.shape
    sets = predict_sets_from_softmax(m, threshold, score_type="aps", tie_rng=tie_rng)
    top = np.argmax(m, axis=1)
    hits, seen = np.zeros(k), np.zeros(k)
    for i, included in enumerate(sets):
        inc = np.zeros(k, dtype=bool)
        inc[included] = True
        others = np.arange(k) != top[i]
        hits += inc & others
        seen += others
    return hits / seen


def _threshold_taking(j: int, k: int, r: float) -> float:
    return r + (j - 0.5) * (1.0 - r) / (k - 1)


@pytest.mark.parametrize("k", [10, 100])
def test_random_tie_order_includes_every_tied_class_equally_often(k: int) -> None:
    r, j = 0.7, (k - 1) // 2
    m = _tied_rows(4000, k, r, seed=0)
    rates = _inclusion_of_tied_classes(m, _threshold_taking(j, k, r), np.random.default_rng(1))
    assert abs(rates.mean() - j / (k - 1)) < 0.01
    assert np.max(np.abs(rates - j / (k - 1))) < 0.08
    assert abs(np.corrcoef(np.arange(k), rates)[0, 1]) < 0.3


@pytest.mark.parametrize("k", [10, 100])
def test_the_argsort_tie_order_is_the_index_artifact_the_random_order_removes(k: int) -> None:
    """The plant: without a tie generator some classes are always or never included."""
    r, j = 0.7, (k - 1) // 2
    m = _tied_rows(4000, k, r, seed=0)
    rates = _inclusion_of_tied_classes(m, _threshold_taking(j, k, r), None)
    assert rates.max() - rates.min() > 0.9


def test_without_a_tie_generator_the_ported_functions_are_unchanged() -> None:
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(12), size=200)
    y = rng.integers(0, 12, 200)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rank = np.argmax(order == y[:, None], axis=1)
    expected = cum[np.arange(200), rank]
    np.testing.assert_array_equal(aps_scores(p, y, randomized=False), expected)
    sets = predict_sets_from_softmax(p, 0.8, score_type="aps")
    for i in range(200):
        stop = int(np.argmax(cum[i] >= 0.8))
        assert sets[i] == [int(c) for c in order[i, : stop + 1]]


def test_untied_probabilities_keep_their_order_under_the_tie_generator() -> None:
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(30), size=300)
    y = rng.integers(0, 30, 300)
    np.testing.assert_array_equal(
        aps_scores(p, y, randomized=False, tie_rng=np.random.default_rng(9)),
        aps_scores(p, y, randomized=False),
    )
    assert predict_sets_from_softmax(p, 0.9, tie_rng=np.random.default_rng(9)) == \
        predict_sets_from_softmax(p, 0.9)


def _acp_per_class(tie_break: str, k: int = 20) -> np.ndarray:
    ds = make_synthetic_shift(K=k, n_per_class=400, shift=0.0, noise=1.6, seed=3)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    half = len(yt) // 2
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(yt))
    audit, test = perm[:half], perm[half:]
    fitted = ACPCalibrator(tie_break=tie_break).fit(
        xc, yc, target_labeled=(xt[audit], yt[audit]), alpha=0.1, seed=0)
    mask = fitted.predict_set(xt[test])
    return np.asarray(per_class_coverage(mask, yt[test], k), dtype=float)


def test_acp_per_class_coverage_does_not_follow_the_class_index() -> None:
    random_cov = _acp_per_class("random")
    index_cov = _acp_per_class("index")
    assert random_cov.min() > index_cov.min() + 0.2
    assert np.max(random_cov) - np.min(random_cov) < 0.25
    assert abs(np.corrcoef(np.arange(len(random_cov)), random_cov)[0, 1]) < 0.5


def test_acp_random_ties_are_reproducible_and_predict_set_is_pure() -> None:
    ds = make_synthetic_shift(K=8, n_per_class=300, shift=0.5, seed=0)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    a = ACPCalibrator().fit(xc, yc, target_labeled=(xt[:400], yt[:400]), alpha=0.1, seed=7)
    b = ACPCalibrator().fit(xc, yc, target_labeled=(xt[:400], yt[:400]), alpha=0.1, seed=7)
    assert np.array_equal(a.thresholds(), b.thresholds())
    first = a.predict_set(xt)
    assert np.array_equal(first, a.predict_set(xt))
    assert np.array_equal(first, b.predict_set(xt))


def test_acp_keeps_marginal_coverage_with_random_ties() -> None:
    ds = make_synthetic_shift(K=20, n_per_class=400, shift=0.0, noise=1.6, seed=5)
    xc, yc = ds.source_cal()
    xt, yt = ds.target_test()
    fitted = ACPCalibrator().fit(
        xc, yc, target_labeled=(xt[:1000], yt[:1000]), alpha=0.1, seed=0)
    mask = fitted.predict_set(xt[1000:])
    cov = mask[np.arange(len(yt) - 1000), yt[1000:]].mean()
    assert cov >= 0.9 - 0.03


def test_acp_rejects_an_unknown_tie_rule() -> None:
    with pytest.raises(ValueError, match="tie_break"):
        ACPCalibrator(tie_break="stable")


def test_outputs_holding_acp_results_record_the_tie_rule(tmp_path) -> None:
    """A rerun's files must be told apart from the submitted runs' files, which used the index
    rule and carry no tie rule."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from pcpbench.calibrators.acp import DEFAULT_TIE_BREAK
    from pcpbench.runner import run

    assert DEFAULT_TIE_BREAK == "random"
    root = Path(__file__).resolve().parents[1]

    def load(name: str):
        spec = importlib.util.spec_from_file_location(f"_tie_{name}", root / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    cks = load("controlled_k_sweep")
    base = ["--cache", "synthetic", "--k-values", "4", "--subset-seeds", "1",
            "--n-cal-per-class", "20"]
    with_acp, without = tmp_path / "acp.json", tmp_path / "no_acp.json"
    assert cks.main(base + ["--methods", "split,acp", "--out", str(with_acp)]) == 0
    assert cks.main(base + ["--methods", "split", "--out", str(without)]) == 0
    assert json.loads(with_acp.read_text())["config"]["acp_tie_break"] == "random"
    assert "acp_tie_break" not in json.loads(without.read_text())["config"]

    lcs = load("label_complexity_sweep")
    assert lcs.to_json_payload([], [], cache="c", alpha=0.1, seeds=1, m_values=[0],
                               methods=["ppi", "acp"], eval_frac=0.5)["acp_tie_break"] == "random"
    assert "acp_tie_break" not in lcs.to_json_payload(
        [], [], cache="c", alpha=0.1, seeds=1, m_values=[0], methods=["ppi"], eval_frac=0.5)

    ds = make_synthetic_shift(K=6, n_per_class=200, shift=0.5, seed=0)
    assert run(ds, "acp", alpha=0.1, seeds=1)["tie_break"] == "random"
    assert "tie_break" not in run(ds, "split", alpha=0.1, seeds=1)
