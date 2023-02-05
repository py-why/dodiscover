import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from dodiscover.cd import BregmanCDTest, KernelCDTest

seed = 12345


def single_env_scm(n_samples=200, offset=0.0):
    # We construct a SCM where X1 -> Y <- X and Y -> Z
    # so X1 is independent from X, but conditionally dependent
    # given Y or Z
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_samples, 1)) + offset
    X1 = rng.standard_normal((n_samples, 1)) + offset
    Y = X + X1 + 0.1 * rng.standard_normal((n_samples, 1))
    Z = Y + 0.1 * rng.standard_normal((n_samples, 1))

    # create input for the CD test
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
    "cd_estimator",
    [
        KernelCDTest(),
        BregmanCDTest(),
    ],
)
def test_cd_tests_error(cd_estimator):
    x = "x"
    y = "y"

    sample_df = single_env_scm()
    with pytest.raises(ValueError, match="The group col"):
        cd_estimator.test(sample_df, {x}, {y}, group_col="blah")

    with pytest.raises(ValueError, match="The x variables are not all"):
        cd_estimator.test(sample_df, {"blah"}, y_vars={y}, group_col="group")

    with pytest.raises(ValueError, match="The y variables are not all"):
        cd_estimator.test(sample_df, {x}, y_vars={"blah"}, group_col="group")

    # all the group indicators have different values now from 0/1
    sample_df["group"] = sample_df["group"] + 3
    with pytest.raises(RuntimeError, match="Group indications in"):
        cd_estimator.test(sample_df, {x}, {y}, group_col="group")


def test_kernel_cd_errors():
    x = "x"
    y = "y"

    sample_df = single_env_scm()
    with pytest.raises(
        ValueError, match="Both propensity model and propensity estimates are specified"
    ):
        cd_estimator = KernelCDTest(
            propensity_model=RandomForestClassifier(), propensity_est=[0.5, 0.5]
        )
        cd_estimator.test(sample_df, {x}, y_vars={y}, group_col="group")

    with pytest.raises(ValueError, match="There are 3 group pre-defined estimates"):
        cd_estimator = KernelCDTest(propensity_est=np.ones((200, 3)) * 0.5)
        cd_estimator.test(sample_df, {x}, y_vars={y}, group_col="group")

    with pytest.raises(ValueError, match="There are 100 pre-defined estimates"):
        cd_estimator = KernelCDTest(propensity_est=np.ones((100, 2)) * 0.5)
        cd_estimator.test(sample_df, {x}, y_vars={y}, group_col="group")


@pytest.mark.parametrize(
    ["cd_func", "cd_kwargs"],
    [
        [BregmanCDTest, dict()],
        [KernelCDTest, dict()],
        [KernelCDTest, {"propensity_model": RandomForestClassifier()}],
        [KernelCDTest, {"propensity_est": np.ones((200, 2)) * 0.5}],
        [KernelCDTest, {"l2": 1e-3}],
        [KernelCDTest, {"l2": (1e-3, 2e-3)}],
    ],
)
@pytest.mark.parametrize(
    ["df", "env_type"],
    [
        [single_env_scm(), "single"],
        [multi_env_scm(), "multi"],
    ],
)
def test_cd_simulation(cd_func, df, env_type, cd_kwargs):
    """Test conditional discrepancy tests."""
    random_state = 12345
    cd_estimator = cd_func(random_state=random_state, null_reps=50, n_jobs=-1, **cd_kwargs)

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
        _, pvalue = cd_estimator.test(df, {"x"}, {"z"}, group_col=group_col)
        assert pvalue < 0.05
        print(pvalue)
        _, pvalue = cd_estimator.test(df, {"x"}, {"y"}, group_col=group_col)
        assert pvalue < 0.05
        print(pvalue)
        _, pvalue = cd_estimator.test(df, {"x1"}, {"z"}, group_col=group_col)
        assert pvalue < 0.05
        print(pvalue)
