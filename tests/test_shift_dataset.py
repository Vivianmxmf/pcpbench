import numpy as np
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.datasets.shift_dataset import ShiftDataset


def test_synthetic_shapes_and_views():
    ds = make_synthetic_shift(K=6, n_per_class=100, shift=0.5, seed=0)
    assert ds.n_classes == 6
    sc_x, sc_y = ds.source_cal()
    tt_x, tt_y = ds.target_test()
    assert sc_x.shape[1] == 6 and tt_x.shape[1] == 6
    assert len(sc_x) == len(sc_y) > 0 and len(tt_x) == len(tt_y) > 0
    assert set(np.unique(sc_y)).issubset(set(range(6)))


def test_deterministic_same_seed():
    a = make_synthetic_shift(K=5, n_per_class=50, seed=7)
    b = make_synthetic_shift(K=5, n_per_class=50, seed=7)
    assert np.array_equal(a.logits, b.logits) and np.array_equal(a.labels, b.labels)


def test_npz_roundtrip(tmp_path):
    ds = make_synthetic_shift(K=4, n_per_class=30, seed=1)
    p = tmp_path / "d.npz"
    ds.to_npz(p)
    ds2 = ShiftDataset.from_npz(p)
    assert np.array_equal(ds.logits, ds2.logits)
    assert np.array_equal(ds.split, ds2.split)
    assert ds.meta["name"] == ds2.meta["name"]


def test_shift_collapses_some_classes():
    ds = make_synthetic_shift(K=8, n_per_class=200, shift=0.7, seed=0)
    sc_x, sc_y = ds.source_cal()
    tt_x, tt_y = ds.target_test()
    src_acc = (sc_x.argmax(1) == sc_y).mean()
    tgt_acc = (tt_x.argmax(1) == tt_y).mean()
    assert tgt_acc < src_acc
