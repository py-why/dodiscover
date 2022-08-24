from math import frexp

import numpy as np
import pandas as pd
import pytest

from dodiscover.ci import GSquareCITest
from dodiscover.testdata import testdata


def test_g_error():
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    df = pd.DataFrame.from_records(dm)
    with pytest.raises(ValueError, match="data_type"):
        ci_estimator = GSquareCITest(data_type="auto")
        ci_estimator.test(df, {x}, {y}, set(sets[0]), [3, 2, 3, 4, 2])


def test_g_discrete():
    """Test G^2 test for discrete data."""
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="discrete")
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(df, {x}, {y}, set(sets[idx]), [3, 2, 3, 4, 2])
        fr_p = frexp(p)
        fr_a = frexp(testdata.dis_answer[idx])
        assert round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]

    # check error message for number of samples
    dm = np.array([testdata.dis_data]).reshape((2000, 25))
    df = pd.DataFrame.from_records(dm)
    levels = np.ones((25,)) * 3
    sets = [[2, 3, 4, 5, 6, 7]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(df, {x}, {y}, set(sets[0]), levels)


def test_g_binary():
    """Test G^2 test for binary data."""
    dm = np.array([testdata.bin_data]).reshape((5000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="binary")
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(df, {x}, {y}, set(sets[idx]))
        fr_p = frexp(p)
        fr_a = frexp(testdata.bin_answer[idx])
        assert round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]

    # check error message for number of samples
    dm = np.array([testdata.bin_data]).reshape((500, 50))
    df = pd.DataFrame.from_records(dm)
    sets = [[2, 3, 4, 5, 6, 7, 8]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(df, {x}, {y}, set(sets[0]))
