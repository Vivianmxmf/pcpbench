import numpy as np
from pcpbench.calibrators.ppi import _isotonic, _ppi_thr


def test_isotonic_is_nondecreasing():
    y = np.array([3.0, 1.0, 2.0, 0.5, 5.0])
    out = _isotonic(y)
    assert np.all(np.diff(out) >= -1e-9)


def test_ppi_thr_matches_naive_when_pseudo_equals_truth():
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 1, size=200))
    py = np.ones(200, dtype=bool)
    yL = np.ones(50)
    pyL = np.ones(50)
    thr = _ppi_thr(s, py, s[:50], yL, pyL, alpha=0.1, M=50, fallback=1.0)
    assert 0.80 <= thr <= 1.0
