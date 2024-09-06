import numpy as np

from dodiscover.toporder.utils import kernel_width


def test_kernel_width_when_zero_median_pairwise_distances():
    arr = np.zeros((100, 1), dtype=np.int64)
    arr[1] = 1
    assert kernel_width(arr) == 1

def test_kernel_width_when_constant_pairwise_distances():
    arr = np.ones((100, 1), dtype=np.int64)
    assert kernel_width(arr) == 1
