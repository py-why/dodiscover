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


def multi_env_scm():
    n_samples = 100
    df = single_env_scm(n_samples=n_samples)
    df["group"] = 0

    new_df = single_env_scm(n_samples=n_samples, offset=1.5)
    new_df["group"] = 1
    df = pd.concat((df, new_df), axis=0)
    return df


@pytest.mark.parametrize(
    ["df", "env_type"],
    [
        [single_env_scm(), "single"],
        [multi_env_scm(), "multi"],
    ],
)
def test_kernel_cd(df, env_type):
    """Test Fisher Z test for Gaussian data."""
    random_state = 12345
    cd_estimator = KernelCDTest(random_state=random_state, null_reps=100, n_jobs=-1)

    group_col = "group"

    if env_type == "single":
        _, pvalue = cd_estimator.test(df, {"x"}, {"x1"}, group_col=group_col)
        print(pvalue)
        assert pvalue > 0.05
        _, pvalue = cd_estimator.test(df, {"x"}, {"z"}, group_col=group_col)
        print(pvalue)
        assert pvalue > 0.05
        _, pvalue = cd_estimator.test(df, {"x"}, {"y"}, group_col=group_col)
        print(pvalue)
        assert pvalue > 0.05
    elif env_type == "multi":
        _, pvalue = cd_estimator.test(df, {"x"}, {"x1"}, group_col=group_col)
        assert pvalue < 0.05
        print(pvalue)
        _, pvalue = cd_estimator.test(df, {"x"}, {"z"}, group_col=group_col)
        assert pvalue < 0.05
        print(pvalue)
        _, pvalue = cd_estimator.test(df, {"x"}, {"y"}, group_col=group_col)
        assert pvalue < 0.05
        print(pvalue)
