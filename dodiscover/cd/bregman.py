from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from dodiscover.ci.kernel_utils import corrent_matrix, von_neumann_divergence
from dodiscover.typing import Column

from .base import BaseConditionalDiscrepancyTest


class BregmanCDTest(BaseConditionalDiscrepancyTest):
    """Bregman divergence conditional discrepancy test.

    Tests the equality of conditional distributions using a kernel approach
    to estimate Bregman divergences outlined in :footcite:`Yu2020Bregman`.

    Parameters
    ----------
    metric : str, optional
        The kernel metric, by default 'rbf'.
    distance_metric : str, optional
        The distance metric, by default 'euclidean'.
    kwidth : float, optional
        The width of the kernel, by default None, which we will then estimate
        using the default procedure in :func:`dodiscover.ci.kernel_utils.compute_kernel`.
    null_reps : int, optional
        Number of times to sample null distribution, by default 1000.
    n_jobs : int, optional
        Number of CPUs to use, by default None.
    propensity_model : callable, optional
        The propensity model to estimate propensity scores among the groups. If `None`
        (default) will use :class:`sklearn.linear_model.LogisticRegression`. The
        ``propensity_model`` passed in must implement a ``predict_proba`` method in
        order to be used. See https://scikit-learn.org/stable/glossary.html#term-predict_proba
        for more information.
    propensity_est : array-like of shape (n_samples, n_groups,), optional
        The propensity estimates for each group. Must match the cardinality of the
        ``group_col`` in the data passed to ``test`` function. If `None` (default),
        will build a propensity model using the argument in ``propensity_model``.
    random_state : int, optional
        Random seed, by default None.

    Notes
    -----
    Currently only testing among two groups are supported. Therefore ``df[group_col]`` must
    only contain binary indicators and ``propensity_est`` must contain only two columns.
    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        metric: str = "rbf",
        distance_metric: str = "euclidean",
        kwidth: Optional[float] = None,
        null_reps: int = 1000,
        n_jobs: Optional[int] = None,
        propensity_model=None,
        propensity_est=None,
        random_state: Optional[int] = None,
    ) -> None:
        self.metric = metric
        self.distance_metric = distance_metric
        self.kwidth = kwidth
        self.null_reps = null_reps
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.propensity_model = propensity_model
        self.propensity_est = propensity_est

    def test(
        self,
        df: pd.DataFrame,
        group_col: Set[Column],
        y_vars: Set[Column],
        x_vars: Set[Column],
    ) -> Tuple[float, float]:
        # check test input
        self._check_test_input(df, y_vars, group_col, x_vars)
        group_col_var: Column = list(group_col)[0]

        x_cols = list(x_vars)
        y_cols = list(y_vars)
        group_ind = df[group_col_var].to_numpy()
        if set(np.unique(group_ind)) != {0, 1}:
            raise RuntimeError(f"Group indications in {group_col_var} column should be all 1 or 0.")

        # get the X and Y dataset
        X = df[x_cols].to_numpy()
        Y = df[y_cols].to_numpy()

        # We are interested in testing: P_1(y|x) = P_2(y|x)
        # compute the conditional divergence, which is symmetric by construction
        # 1/2 * (D(p_1(y|x) || p_2(y|x)) + D(p_2(y|x) || p_1(y|x)))
        conditional_div = self._statistic(X, Y, group_ind)

        # compute propensity scores
        e_hat = self._compute_propensity_scores(group_ind, K=X)

        # now compute null distribution
        null_dist = self.compute_null(
            e_hat, X, Y, null_reps=self.null_reps, random_state=self.random_state
        )
        self.null_dist_ = null_dist

        # compute pvalue
        pvalue = (1.0 + np.sum(null_dist >= conditional_div)) / (1 + self.null_reps)
        return conditional_div, pvalue

    def _statistic(self, X: ArrayLike, Y: ArrayLike, group_ind: ArrayLike) -> float:
        first_group = group_ind == 0
        second_group = group_ind == 1
        X1 = X[first_group, :]
        X2 = X[second_group, :]
        Y1 = Y[first_group, :]
        Y2 = Y[second_group, :]

        # first compute the centered correntropy matrices, C_xy^1
        Cx1y1 = corrent_matrix(np.hstack((X1, Y1)), kwidth=self.kwidth)
        Cx2y2 = corrent_matrix(np.hstack((X2, Y2)), kwidth=self.kwidth)

        # compute the centered correntropy matrices for just C_x^1 and C_x^2
        Cx1 = corrent_matrix(
            X1,
            metric=self.metric,
            distance_metric=self.distance_metric,
            kwidth=self.kwidth,
            n_jobs=self.n_jobs,
        )
        Cx2 = corrent_matrix(
            X2,
            metric=self.metric,
            distance_metric=self.distance_metric,
            kwidth=self.kwidth,
            n_jobs=self.n_jobs,
        )

        # compute the conditional divergence with the Von Neumann div
        # D(p_1(y|x) || p_2(y|x))
        joint_div1 = von_neumann_divergence(Cx1y1, Cx2y2)
        joint_div2 = von_neumann_divergence(Cx2y2, Cx1y1)
        x_div1 = von_neumann_divergence(Cx1, Cx2)
        x_div2 = von_neumann_divergence(Cx2, Cx1)

        # compute the conditional divergence, which is symmetric by construction
        # 1/2 * (D(p_1(y|x) || p_2(y|x)) + D(p_2(y|x) || p_1(y|x)))
        conditional_div = 1.0 / 2 * (joint_div1 - x_div1 + joint_div2 - x_div2)
        return conditional_div
