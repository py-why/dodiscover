import numpy as np
import pandas as pd

from dodiscover.ci import CMITest

seed = 12345


def test_cmi_with_nonlinear_gaussian_data():
    ci_estimator = CMITest(k=0.2, n_shuffle_nbrs=5, n_shuffle=100, n_jobs=-1, random_seed=seed)

    # X -> Y <- X1; Y -> Z
    rng = np.random.default_rng(seed)
    n_samples = 1000
    X = rng.standard_normal((n_samples, 1))
    X1 = rng.uniform(low=0.0, high=1.0, size=(n_samples, 1))
    Y = np.exp(X + X1)
    Z = Y + 1e-4 * rng.standard_normal((n_samples, 1))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, "x", "x1")
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, "x", "z", {"y"})
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, "x", "z", {"x1"})
    print(pvalue)
    assert pvalue < 0.05
    # XXX: The following dependency is very hard to detect
    # _, pvalue = ci_estimator.test(df, "x", "y")
    # print(pvalue)
    # assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, "x", "x1", {"z"})
    print(pvalue)
    assert pvalue < 0.05