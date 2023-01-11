from abc import ABCMeta, abstractmethod
from typing import Set, Tuple

import pandas as pd

from dodiscover.typing import Column


class BaseConditionalDiscrepancyTest(metaclass=ABCMeta):
    """Abstract class for any conditional discrepancy test.

    All CD tests are used in constraint-based causal discovery algorithms. This
    class interface is expected to be very lightweight to enable anyone to convert
    a function for CD testing into a class, which has a specific API.
    """

    def _check_test_input(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        group_col: Column,
    ):
        if any(col not in df.columns for col in x_vars):
            raise ValueError("The x variables are not all in the DataFrame.")
        if any(col not in df.columns for col in y_vars):
            raise ValueError("The y variables are not all in the DataFrame.")
        if group_col not in df.columns:
            raise ValueError(f"The group column {group_col} is not in the DataFrame.")

    @abstractmethod
    def test(
        self, df: pd.DataFrame, x_vars: Set[Column], y_vars: Set[Column], group_col: Column
    ) -> Tuple[float, float]:
        """Abstract method for all conditional discrepancy tests.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        x_vars : Set of column
            A column in ``df``.
        y_vars : Set of column
            A column in ``df``.
        group_col : column
            A column in ``df`` that indicates which group of distribution
            each sample belongs to with a '0', or '1'.

        Returns
        -------
        Tuple[float, float]
            Test statistic and pvalue.
        """
        pass
