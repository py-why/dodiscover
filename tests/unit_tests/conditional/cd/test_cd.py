import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from dodiscover.cd import BregmanCDTest, KernelCDTest

seed = 12345

# number of samples to use in generating test dataset; the lower the faster
n_samples = 160


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


def multi_env_scm(n_samples=100, offset=1.5):
    df = single_env_scm(n_samples=n_samples)
    df["group"] = 0

    new_df = single_env_scm(n_samples=n_samples, offset=offset)
    new_df["group"] = 1
    df = pd.concat((df, new_df), axis=0)
    return df


@pytest.mark.parametrize(
    "cd_func",
    [
        KernelCDTest,
        BregmanCDTest,
    ],
)
def test_cd_tests_error(cd_func):
    x = "x"
    y = "y"

    sample_df = single_env_scm(n_samples=10)
    cd_estimator = cd_func()
    with pytest.raises(ValueError, match="The group col"):
        cd_estimator.test(sample_df, y_vars={y}, group_col={"blah"}, x_vars={x})

    with pytest.raises(ValueError, match="The x variables are not all"):
        cd_estimator.test(sample_df, y_vars={y}, group_col={"group"}, x_vars={"blah"})

    with pytest.raises(ValueError, match="The y variables are not all"):
        cd_estimator.test(sample_df, y_vars={"blah"}, group_col={"group"}, x_vars={x})

    with pytest.raises(ValueError, match="Group column should be only one column"):
        cd_estimator.test(sample_df, y_vars={"blah"}, group_col="group", x_vars={x})

    # all the group indicators have different values now from 0/1
    sample_df["group"] = sample_df["group"] + 3
    with pytest.raises(RuntimeError, match="Group indications in"):
        cd_estimator.test(sample_df, y_vars={y}, group_col={"group"}, x_vars={x})

    # test pre-fit propensity scores, or custom propensity model
    with pytest.raises(
        ValueError, match="Both propensity model and propensity estimates are specified"
    ):
        cd_estimator = cd_func(propensity_model=RandomForestClassifier(), propensity_est=[0.5, 0.5])
        cd_estimator.test(sample_df, y_vars={y}, group_col={"group"}, x_vars={x})

    with pytest.raises(ValueError, match="There are 3 group pre-defined estimates"):
        cd_estimator = cd_func(propensity_est=np.ones((10, 3)) * 0.5)
        cd_estimator.test(sample_df, y_vars={y}, group_col={"group"}, x_vars={x})

    with pytest.raises(ValueError, match="There are 100 pre-defined estimates"):
        cd_estimator = cd_func(propensity_est=np.ones((100, 2)) * 0.5)
        cd_estimator.test(sample_df, y_vars={y}, group_col={"group"}, x_vars={x})


@pytest.mark.parametrize(
    ["cd_func", "cd_kwargs"],
    [
        [BregmanCDTest, dict()],
        [KernelCDTest, dict()],
        [BregmanCDTest, {"propensity_model": RandomForestClassifier()}],
        [BregmanCDTest, {"propensity_est": np.ones((n_samples, 2)) * 0.5}],
        [KernelCDTest, {"propensity_model": RandomForestClassifier()}],
        [KernelCDTest, {"propensity_est": np.ones((n_samples, 2)) * 0.5}],
        [KernelCDTest, {"l2": 1e-3}],
        [KernelCDTest, {"l2": (1e-3, 2e-3)}],
    ],
)
@pytest.mark.parametrize(
    ["df", "env_type"],
    [
        [single_env_scm(n_samples=n_samples, offset=2.0), "single"],
        [multi_env_scm(n_samples=n_samples // 2, offset=2.0), "multi"],
    ],
)
def test_cd_simulation(cd_func, df, env_type, cd_kwargs):
    """Test conditional discrepancy tests."""
    random_state = 12345
    cd_estimator = cd_func(random_state=random_state, null_reps=15, n_jobs=-1, **cd_kwargs)

    group_col = "group"
    alpha = 0.1

    if env_type == "single":
        _, pvalue = cd_estimator.test(
            df,
            y_vars={"x1"},
            group_col={group_col},
            x_vars={"x"},
        )
        assert pvalue > alpha, f"Fails with {pvalue} not greater than {alpha}"
        _, pvalue = cd_estimator.test(df, y_vars={"z"}, group_col={group_col}, x_vars={"x"})
        assert pvalue > alpha, f"Fails with {pvalue} not greater than {alpha}"
        _, pvalue = cd_estimator.test(df, y_vars={"y"}, group_col={group_col}, x_vars={"x"})
        assert pvalue > alpha, f"Fails with {pvalue} not greater than {alpha}"
    elif env_type == "multi":
        _, pvalue = cd_estimator.test(df, y_vars={"z"}, group_col={group_col}, x_vars={"x"})
        assert pvalue < alpha, f"Fails with {pvalue} not less than {alpha}"
        _, pvalue = cd_estimator.test(df, y_vars={"y"}, group_col={group_col}, x_vars={"x"})
        assert pvalue < alpha, f"Fails with {pvalue} not less than {alpha}"
        _, pvalue = cd_estimator.test(df, y_vars={"z"}, group_col={group_col}, x_vars={"x1"})
        assert pvalue < alpha, f"Fails with {pvalue} not less than {alpha}"
