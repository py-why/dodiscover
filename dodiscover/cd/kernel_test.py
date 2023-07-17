from typing import Set, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from dodiscover.ci.kernel_utils import _default_regularization, compute_kernel
from dodiscover.typing import Column

from .base import BaseConditionalDiscrepancyTest


class KernelCDTest(BaseConditionalDiscrepancyTest):
    """Kernel conditional discrepancy test among conditional distributions.

    Tests the equality of conditional distributions using a kernel approach
    outlined in :footcite:`Park2021conditional`.

    Parameters
    ----------
    distance_metric : str, optional
        Distance metric to use, by default "euclidean". For others, see
        :class:`~sklearn.metrics.DistanceMetric` for supported list of metrics.
    metric : str, optional
        Kernel metric, by default "rbf". For others, see :mod:`~sklearn.metrics.pairwise`
        for supported kernel metrics.
    l2 : float | tuple of float, optional
        The l2 regularization to apply for inverting the kernel matrices of 'x' and 'y'
        respectively, by default None. If a single number, then the same l2 regularization
        will be applied to inverting both matrices. If ``None``, then a default
        regularization will be computed that chooses the value that minimizes the upper bound
        on the mean squared prediction error.
    kwidth_x : float, optional
        Kernel width among X variables, by default None, which we will then estimate
        using the median l2 distance between the X variables.
    kwidth_y : float, optional
        Kernel width among Y variables, by default None, which we will then estimate
        using the median l2 distance between the Y variables.
    null_reps : int, optional
        Number of times to sample the null distribution, by default 1000.
    n_jobs : int, optional
        Number of jobs to run computations in parallel, by default None.
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
        distance_metric="euclidean",
        metric="rbf",
        l2=None,
        kwidth_x=None,
        kwidth_y=None,
        null_reps: int = 1000,
        n_jobs=None,
        propensity_model=None,
        propensity_est=None,
        random_state=None,
    ) -> None:
        self.l2 = l2
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.null_reps = null_reps

        self.kwidth_x = kwidth_x
        self.kwidth_y = kwidth_y
        self.metric = metric
        self.distance_metric = distance_metric

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

        # compute kernel for the X and Y data
        X = df[x_cols].to_numpy()
        Y = df[y_cols].to_numpy()
        K, sigma_x = compute_kernel(
            X,
            distance_metric=self.distance_metric,
            metric=self.metric,
            kwidth=self.kwidth_x,
            n_jobs=self.n_jobs,
        )
        L, sigma_y = compute_kernel(
            Y,
            distance_metric=self.distance_metric,
            metric=self.metric,
            kwidth=self.kwidth_y,
            n_jobs=self.n_jobs,
        )

        # store fitted attributes
        self.kwidth_x_ = sigma_x
        self.kwidth_y_ = sigma_y

        # compute the statistic
        stat = self._statistic(K, L, group_ind)

        # compute propensity scores
        e_hat = self._compute_propensity_scores(group_ind, K=K)

        # now compute null distribution
        null_dist = self.compute_null(
            e_hat, K, L, null_reps=self.null_reps, random_state=self.random_state
        )
        self.null_dist_ = null_dist

        # compute the pvalue
        pvalue = (1 + np.sum(null_dist >= stat)) / (1 + self.null_reps)
        return stat, pvalue

    def _statistic(self, K: ArrayLike, L: ArrayLike, group_ind: ArrayLike) -> float:
        n_samples = len(K)

        # compute W matrices from K and z
        W0, W1 = self._compute_inverse_kernel(K, group_ind)

        # compute L kernels
        first_mask = np.array(1 - group_ind, dtype=bool)
        second_mask = np.array(group_ind, dtype=bool)
        L0 = L[np.ix_(first_mask, first_mask)]
        L1 = L[np.ix_(second_mask, second_mask)]
        L01 = L[np.ix_(first_mask, second_mask)]

        # compute the final test statistic
        K0 = K[:, first_mask]
        K1 = K[:, second_mask]
        KW0 = K0 @ W0
        KW1 = K1 @ W1

        # compute the three terms in Lemma 4.4
        first_term = np.trace(KW0.T @ KW0 @ L0)
        second_term = np.trace(KW1.T @ KW0 @ L01)
        third_term = np.trace(KW1.T @ KW1 @ L1)

        # compute final statistic
        stat = (first_term - 2 * second_term + third_term) / n_samples
        return stat

    def _compute_inverse_kernel(self, K, z) -> Tuple[ArrayLike, ArrayLike]:
        """Compute W matrices as done in KCD test.

        Parameters
        ----------
        K : ArrayLike of shape (n_samples, n_samples)
            The kernel matrix.
        z : ArrayLike of shape (n_samples)
            The indicator variable of 1's and 0's for which samples belong
            to which group.

        Returns
        -------
        W0 : ArrayLike of shape (n_samples_i, n_samples_i)
            The inverse of the kernel matrix from the first group.
        W1 : NDArraArrayLike of shape (n_samples_j, n_samples_j)
            The inverse of the kernel matrix from the second group.

        Notes
        -----
        Compute the W matrix for the estimated conditional average in
        the KCD test :footcite:`Park2021conditional`.

        References
        ----------
        .. footbibliography::
        """
        # compute kernel matrices
        first_mask = np.array(1 - z, dtype=bool)
        second_mask = np.array(z, dtype=bool)

        # TODO: CHECK THAT THIS IS CORRECT
        K0 = K[np.ix_(first_mask, first_mask)]
        K1 = K[np.ix_(second_mask, second_mask)]

        # compute regularization factors
        self._get_regs(K0, K1)

        # compute the number of samples in each
        n0 = int(np.sum(1 - z))
        n1 = int(np.sum(z))

        W0 = np.linalg.inv(K0 + self.regs_[0] * np.identity(n0))
        W1 = np.linalg.inv(K1 + self.regs_[1] * np.identity(n1))
        return W0, W1

    def _get_regs(self, K0: ArrayLike, K1: ArrayLike):
        """Compute regularization factors."""
        if isinstance(self.l2, (int, float)):
            l0 = self.l2
            l1 = self.l2
            self.regs_ = (l0, l1)
        elif self.l2 is None:
            self.regs_ = (_default_regularization(K0), _default_regularization(K1))
        else:
            if len(self.l2) != 2:
                raise RuntimeError(f"l2 regularization {self.l2} must be a 2-tuple, or a number.")
            self.regs_ = self.l2
