from abc import ABCMeta, abstractmethod
from typing import Optional, Set, Tuple

import pandas as pd

from dodiscover.typing import Column


class BaseConditionalIndependenceTest(metaclass=ABCMeta):
    """Abstract class for any conditional independence test.

    All CI tests are used in constraint-based causal discovery algorithms. This
    class interface is expected to be very lightweight to enable anyone to convert
    a function for CI testing into a class, which has a specific API.
    """

    # default CI tests do not allow multivariate input
    _allow_multivariate_input: bool = False

    def _check_test_input(self, df: pd.DataFrame, x_vars, y_vars, z_covariates):
        if any(col not in df.columns for col in [x_vars, y_vars]):
            raise ValueError("The x and y variables are not both in the DataFrame.")
        if z_covariates is not None and any(col not in df.columns for col in z_covariates):
            raise ValueError("The z conditioning set variables are not all in the DataFrame.")

        if not self._allow_multivariate_input and (len(x_vars) > 1 or len(y_vars) > 1):
            raise RuntimeError(f"{self.__class__} does not support multivariate input for X and Y.")

    @abstractmethod
    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        """Abstract method for all conditional independence tests.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        x_vars : Set of column
            A column in ``df``.
        y_vars : Set of column
            A column in ``df``.
        z_covariates : Set, optional
            A set of columns in ``df``, by default None. If None, then
            the test should run a standard independence test.

        Returns
        -------
        Tuple[float, float]
            Test statistic and pvalue.
        """
        pass
