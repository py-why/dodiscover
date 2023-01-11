import numpy as np
import pandas as pd
import pytest

from dodiscover.cd import KernelCDTest

seed = 12345


def single_env_scm(n_samples=200, offset=0.0):
    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_samples, 1)) + offset
    X1 = rng.standard_normal((n_samples, 1)) + offset
    Y = X + X1 + 0.5 * rng.standard_normal((n_samples, 1))
    Z = Y + 0.1 * rng.standard_normal((n_samples, 1))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    # assign groups randomly
    df["group"] = rng.choice([0, 1], size=len(df))
    return df


@pytest.mark.parametrize(
    "cd_estimator",
    [
        KernelCDTest(),
    ],
)
def test_ci_tests(cd_estimator):
    x = "x"
    y = "y"

    sample_df = single_env_scm()
    with pytest.raises(ValueError, match="The group col"):
        cd_estimator.test(sample_df, {x}, {y}, group_col="blah")

    with pytest.raises(ValueError, match="The x variables are not all"):
        cd_estimator.test(sample_df, {"blah"}, y_vars={y}, group_col="group")

    with pytest.raises(ValueError, match="The y variables are not all"):
        cd_estimator.test(sample_df, {x}, y_vars={"blah"}, group_col="group")
