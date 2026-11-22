import numpy as np
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.calibrators.split import SplitCP
from pcpbench.calibrators.weighted import WeightedCP


def test_weighted_recovers_marginal_under_shift():
    ds = make_synthetic_shift(K=6, n_per_class=500, shift=0.6, seed=0)
    xc, yc = ds.source_cal(); xt, yt = ds.target_test()
    split_marg = SplitCP().fit(xc, yc, alpha=0.1).predict_set(xt)[np.arange(len(yt)), yt].mean()
    wmask = WeightedCP().fit(xc, yc, target_logits=xt, alpha=0.1, seed=0).predict_set(xt)
    wmarg = wmask[np.arange(len(yt)), yt].mean()
    assert wmarg >= split_marg - 0.02


def test_weighted_predict_set_independent_of_fit_target_order():
    ds = make_synthetic_shift(K=6, n_per_class=500, shift=0.6, seed=0)
    xc, yc = ds.source_cal(); xt, _ = ds.target_test()

    fitted = WeightedCP().fit(xc, yc, target_logits=xt, alpha=0.1, seed=0)
    mask_unshuffled = fitted.predict_set(xt)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(xt))
    xt_shuffled = xt[perm]

    mask_shuffled = fitted.predict_set(xt_shuffled)

    assert mask_shuffled.shape == mask_unshuffled.shape
    np.testing.assert_array_equal(mask_shuffled, mask_unshuffled[perm])
