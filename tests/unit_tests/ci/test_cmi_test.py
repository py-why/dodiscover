import numpy as np
import pandas as pd

from dodiscover.ci import CMITest

seed = 12345


def test_cmi_with_nonlinear_gaussian_data():
    ci_estimator = CMITest(k=0.2, n_shuffle_nbrs=10)

    # X -> Y <- X1; Y -> Z
    rng = np.random.RandomState(seed)
    n_samples = 500
    X = np.power(rng.randn(n_samples, 1), 2)
    X1 = rng.randn(n_samples, 1)
    Y = np.exp(X * X1 + 0.1 * rng.randn(n_samples, 1))
    Z = np.exp(Y + 0.1 * rng.randn(n_samples, 1))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, "x", "x1")
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, "x", "z", {"y"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, "x", "z", {"x1"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, "x", "z")
    assert pvalue < 0.05
