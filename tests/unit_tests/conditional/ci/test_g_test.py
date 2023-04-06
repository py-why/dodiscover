from math import frexp

import numpy as np
import pandas as pd
import pytest

from dodiscover.ci import GSquareCITest
from dodiscover.testdata import testdata

seed = 12345


def test_g_error():
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    df = pd.DataFrame.from_records(dm)
    with pytest.raises(ValueError, match="data_type"):
        ci_estimator = GSquareCITest(data_type="auto", levels=[3, 2, 3, 4, 2])
        ci_estimator.test(df, {x}, {y}, set(sets[0]))


def test_g_discrete():
    """Test G^2 test for discrete data."""
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="discrete", levels=[3, 2, 3, 4, 2])
    df = pd.DataFrame.from_records(dm)

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(df, {x}, {y}, set(sets[idx]))
        fr_p = frexp(p)
        fr_a = frexp(testdata.dis_answer[idx])

        # due to adding small perturbation to prevent dividing by 0 within
        # G^2 statistic computation
        assert round(fr_p[0] - fr_a[0], 3) == 0
        assert fr_p[1] == fr_a[1]
        assert fr_p[0] > 0

    # check error message for number of samples
    dm = np.array([testdata.dis_data]).reshape((2000, 25))
    df = pd.DataFrame.from_records(dm)
    levels = np.ones((25,)) * 3
    ci_estimator = GSquareCITest(data_type="discrete", levels=levels)
    sets = [[2, 3, 4, 5, 6, 7]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(df, {x}, {y}, set(sets[0]))


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
        assert fr_p[1] == fr_a[1]
        assert round(fr_p[0] - fr_a[0], 4) == 0
        assert fr_p[0] > 0

    # check error message for number of samples
    dm = np.array([testdata.bin_data]).reshape((500, 50))
    df = pd.DataFrame.from_records(dm)
    sets = [[2, 3, 4, 5, 6, 7, 8]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(df, {x}, {y}, set(sets[0]))


def binary_scm(n_samples=200):
    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    rng = np.random.default_rng(seed)

    X = rng.binomial(1, 0.3, (n_samples, 1))
    X1 = rng.binomial(1, 0.6, (n_samples, 1))
    Y = X * X1
    Z = Y + (1 - rng.binomial(1, 0.5, (n_samples, 1)))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])
    return df


def test_g_binary_simulation():
    """Test G^2 test for binary data."""
    rng = np.random.default_rng(seed)
    n_samples = 500
    df = binary_scm(n_samples=n_samples)
    for i in range(10):
        df[i] = rng.binomial(1, p=0.5, size=n_samples)
    ci_estimator = GSquareCITest(data_type="binary")

    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x1"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x1"}, {0})
    assert pvalue > 0.05

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, {"y"})
    assert pvalue < 0.05


def test_g_binary_highdim():
    """Test G^2 test for binary data."""
    rng = np.random.default_rng(seed)
    n_samples = 1000
    df = binary_scm(n_samples=n_samples)
    for i in range(10):
        df[i] = rng.binomial(1, p=0.8, size=n_samples)
    ci_estimator = GSquareCITest(data_type="binary")

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, set(range(6)))
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"}, set(range(5)).union({"x1"}))
    assert pvalue < 0.05
