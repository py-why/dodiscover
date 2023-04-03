import numpy as np
import pandas as pd
import pytest

from dodiscover.ci import CMITest

seed = 12345


@pytest.mark.parametrize("transform", ["rank", "standardize", "uniform"])
def test_cmi_with_nonlinear_gaussian_data(transform):
    ci_estimator = CMITest(
        k=0.2, n_shuffle_nbrs=5, n_shuffle=25, transform=transform, n_jobs=-1, random_seed=seed
    )

    # X -> Y <- X1; Y -> Z
    rng = np.random.default_rng(seed)
    n_samples = 100
    X = rng.standard_normal((n_samples, 1))
    X1 = rng.uniform(low=0.0, high=1.0, size=(n_samples, 1))
    Y = np.exp((X + X1) / 10)
    Z = Y + 1e-4 * rng.standard_normal((n_samples, 1))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x1"}, {"z"}, {"y", "x"})
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"x1"})
    print(pvalue)
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    print(pvalue)
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    print(pvalue)
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, {"z"})
    print(pvalue)
    assert pvalue < 0.05
