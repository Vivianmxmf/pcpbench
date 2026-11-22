import numpy as np
from pcpbench.metrics.coverage import (marginal_coverage, per_class_coverage,
    worst_class_coverage, frac_classes_undercovered, mean_set_size)


def test_metrics_on_handbuilt_mask():
    mask = np.array([[1,0,0],[1,1,0],[0,0,1],[1,0,0]], bool)
    y = np.array([0,1,2,1])
    assert marginal_coverage(mask, y) == 3/4
    pc = per_class_coverage(mask, y, 3)
    assert pc[0] == 1.0 and pc[1] == 0.5 and pc[2] == 1.0
    assert worst_class_coverage(mask, y, 3) == 0.5
    assert frac_classes_undercovered(mask, y, 3, level=0.9) == 1/3
    assert mean_set_size(mask) == (1+2+1+1)/4
