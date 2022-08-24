from math import log, sqrt
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from dodiscover.typing import Column

from .base import BaseConditionalIndependenceTest


class FisherZCITest(BaseConditionalIndependenceTest):
    def __init__(self, correlation_matrix=None):
        """Conditional independence test using Fisher-Z's test for Gaussian random variables.

        Parameters
        ----------
        correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
            ``None`` means without the parameter of correlation matrix and
            the correlation will be computed from the data., by default None
        """
        self.correlation_matrix = correlation_matrix

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
        stat : float
            The test statistic.
        pvalue : float
            The p-value of the test.
        """
        self._check_test_input(df, x_vars, y_vars, z_covariates)
        if z_covariates is None:
            z_covariates = set()
        x_var = x_vars.pop()
        y_var = y_vars.pop()

        stat, pvalue = fisherz(df, x_var, y_var, z_covariates, self.correlation_matrix)
        return stat, pvalue


def fisherz(
    data: pd.DataFrame,
    x: Column,
    y: Column,
    sep_set: Set,
    correlation_matrix=None,
):
    """Perform an independence test using Fisher-Z's test.

    Works on Gaussian random variables.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    x : Column
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : Column
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).
    correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
            ``None`` means without the parameter of correlation matrix and
            the correlation will be computed from the data., by default None

    Returns
    -------
    X : float
        The test statistic.
    p : float
        The p-value of the test.
    """
    data_arr = data.to_numpy()

    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data_arr.T)
    sample_size = data.shape[0]
    var = list({x, y}.union(sep_set))  # type: ignore
    (var_idx,) = np.in1d(data.columns, var).nonzero()

    # compute the correlation matrix within the specified data
    sub_corr_matrix = correlation_matrix[np.ix_(var_idx, var_idx)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])

    # apply the Fisher Z-transformation
    Z = 0.5 * log((1 + r) / (1 - r))

    # compute the test statistic
    X = sqrt(sample_size - len(sep_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    return X, p
