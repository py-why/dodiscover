import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from dodiscover.ci import CategoricalCITest

df_adult = pd.read_csv("dodiscover/testdata/adult.csv")


def test_chisquare_marginal_independence_adult_dataset():
    """Test that chi-square tests return the correct answer for marginal independence queries.

    Uses the test data from dagitty.
    """
    # Comparision values taken from dagitty (DAGitty)
    ci_est = CategoricalCITest("pearson")
    coef, p_value = ci_est.test(x_vars={"Age"}, y_vars={"Immigrant"}, z_covariates=[], df=df_adult)
    assert_almost_equal(coef, 57.75, decimal=1)
    assert_almost_equal(np.log(p_value), -25.47, decimal=1)
    assert ci_est.dof_ == 4

    coef, p_value = ci_est.test(x_vars={"Age"}, y_vars={"Race"}, z_covariates=[], df=df_adult)
    assert_almost_equal(coef, 56.25, decimal=1)
    assert_almost_equal(np.log(p_value), -24.75, decimal=1)
    assert ci_est.dof_ == 4

    coef, p_value = ci_est.test(x_vars={"Age"}, y_vars={"Sex"}, z_covariates=[], df=df_adult)
    assert_almost_equal(coef, 289.62, decimal=1)
    assert_almost_equal(np.log(p_value), -139.82, decimal=1)
    assert ci_est.dof_ == 4

    coef, p_value = ci_est.test(x_vars={"Immigrant"}, y_vars={"Sex"}, z_covariates={}, df=df_adult)
    assert_almost_equal(coef, 0.2724, decimal=1)
    assert_almost_equal(np.log(p_value), -0.50, decimal=1)
    assert ci_est.dof_ == 1


def test_chisquare_conditional_independence_adult_dataset():
    """Test that chi-square tests return the correct answer for conditional independence queries.

    Uses the test data from dagitty.
    """
    ci_est = CategoricalCITest("pearson")

    coef, p_value = coef, p_value = ci_est.test(
        x_vars={"Education"},
        y_vars={"HoursPerWeek"},
        z_covariates=["Age", "Immigrant", "Race", "Sex"],
        df=df_adult,
    )
    assert_almost_equal(coef, 1460.11, decimal=1)
    assert_almost_equal(p_value, 0, decimal=1)
    assert ci_est.dof_ == 316

    coef, p_value = ci_est.test(
        x_vars={"Education"}, y_vars={"MaritalStatus"}, z_covariates=["Age", "Sex"], df=df_adult
    )
    assert_almost_equal(coef, 481.96, decimal=1)
    assert_almost_equal(p_value, 0, decimal=1)
    assert ci_est.dof_ == 58

    # Values differ (for next 2 tests) from dagitty because dagitty ignores grouped
    # dataframes with very few samples. Update: Might be same from scipy_vars=1.7.0
    coef, p_value = ci_est.test(
        x_vars={"Income"},
        y_vars={"Race"},
        z_covariates=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
        df=df_adult,
    )

    assert_almost_equal(coef, 66.39, decimal=1)
    assert_almost_equal(p_value, 0.99, decimal=1)
    assert ci_est.dof_ == 136

    coef, p_value = ci_est.test(
        x_vars={"Immigrant"},
        y_vars={"Income"},
        z_covariates=["Age", "Education", "HoursPerWeek", "MaritalStatus"],
        df=df_adult,
    )
    assert_almost_equal(coef, 65.59, decimal=1)
    assert_almost_equal(p_value, 0.999, decimal=2)
    assert ci_est.dof_ == 131


@pytest.mark.parametrize(
    "ci_test",
    [
        CategoricalCITest("pearson"),  # chi-square
        CategoricalCITest("log-likelihood"),  # G^2
        CategoricalCITest("freeman-tukey"),  # freeman-tukey
        CategoricalCITest("mod-log-likelihood"),  # Modified log-likelihood
        CategoricalCITest("neyman"),  # Neyman
        CategoricalCITest("cressie-read"),  # Cressie-read
    ],
)
def test_chisquare_when_dependent(ci_test):
    assert (
        ci_test.test(
            x_vars={"Age"},
            y_vars={"Immigrant"},
            z_covariates=[],
            df=df_adult,
        )[1]
        < 0.05
    )

    assert (
        ci_test.test(
            x_vars={"Age"},
            y_vars={"Race"},
            z_covariates=[],
            df=df_adult,
        )[1]
        < 0.05
    )

    assert (
        ci_test.test(
            x_vars={"Age"},
            y_vars={"Sex"},
            z_covariates=[],
            df=df_adult,
        )[1]
        < 0.05
    )
    assert (
        ci_test.test(
            x_vars={"Immigrant"},
            y_vars={"Sex"},
            z_covariates=[],
            df=df_adult,
        )[1]
        >= 0.05
    )

    assert (
        ci_test.test(
            x_vars={"Education"},
            y_vars={"HoursPerWeek"},
            z_covariates=["Age", "Immigrant", "Race", "Sex"],
            df=df_adult,
        )[1]
        < 0.05
    )
    assert (
        ci_test.test(
            x_vars={"Education"},
            y_vars={"MaritalStatus"},
            z_covariates=["Age", "Sex"],
            df=df_adult,
        )[1]
        < 0.05
    )


@pytest.mark.parametrize(
    "ci_test",
    [
        CategoricalCITest("pearson"),  # chi-square
        CategoricalCITest("log-likelihood"),  # G^2
        CategoricalCITest("freeman-tukey"),  # freeman-tukey
        CategoricalCITest("mod-log-likelihood"),  # Modified log-likelihood
        CategoricalCITest("neyman"),  # Neyman
        CategoricalCITest("cressie-read"),  # Cressie-read
    ],
)
def test_chisquare_when_exactly_dependent(ci_test):
    x = np.random.choice([0, 1], size=1000)
    y = x.copy()
    df = pd.DataFrame({"x": x, "y": y})

    stat, p_value = ci_test.test(x_vars={"x"}, y_vars={"y"}, z_covariates=[], df=df)
    assert ci_test.dof_ == 1
    assert_almost_equal(p_value, 0, decimal=5)
