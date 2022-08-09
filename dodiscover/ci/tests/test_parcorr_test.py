import numpy as np
import pandas as pd
import pytest

from causal_networkx.ci import PartialCorrelation

seed = 12345

ci_params = {
    "parcorr_analytic": PartialCorrelation(),
    "parcorr_fixed_threshold": PartialCorrelation(method="fixed_threshold"),
    "pingouin": PartialCorrelation(method="pingouin"),
    # "parcorr_shuffle": PartialCorrelation(method="shuffle_test", bootstrap_n_samples=1000),
}


@pytest.mark.parametrize("ci_estimator", ci_params.values(), ids=ci_params.keys())
def test_parcorr_with_gaussian_data(ci_estimator):
    # ci_estimator = PartialCorrelation(method='shuffle_test')
    n_samples = 300

    # X -> Y -> Z with W as an independent node
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 1)
    W = rng.randn(n_samples, 1)
    Y = X + 0.5 * rng.randn(n_samples, 1)
    Z = Y + 0.5 * rng.randn(n_samples, 1)

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, W, Y, Z)), columns=["x", "w", "y", "z"])

    _, pvalue = ci_estimator.test(df, "x", "w")
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, "x", "z")
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, "x", "z", "y")
    assert pvalue > 0.05

    # _, pvalue = ci_estimator.test(X, Z, np.hstack((Y, W)))
    # assert pvalue > 0.05
