from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import pandas as pd


class BaseConditionalIndependenceTest(metaclass=ABCMeta):
    """Abstract class for any conditional independence test.

    All CI tests are used in constraint-based causal discovery algorithms. This
    class interface is expected to be very lightweight to enable anyone to convert
    a function for CI testing into a class, which has a specific API.
    """

    def _check_test_input(self, df: pd.DataFrame, x_var, y_var, z_covariates):
        if any(col not in df.columns for col in [x_var, y_var]):
            raise ValueError("The x and y variables are not both in the DataFrame.")
        if z_covariates is not None and any(col not in df.columns for col in z_covariates):
            raise ValueError("The z conditioning set variables are not all in the DataFrame.")

    @abstractmethod
    def test(
        self, df: pd.DataFrame, x_var: Any, y_var: Any, z_covariates: Any = None
    ) -> Tuple[float, float]:
        """Abstract method for all conditional independence tests.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        x_var : Any
            A column in `df`.
        y_var : Any
            A column in `df`.
        z_covariates : Any, optional
            A set of columns in `df`, by default None. If None, then
            the test should run a standard independence test.

        Returns
        -------
        Tuple[float, float]
            Test statistic and pvalue.
        """
        pass
