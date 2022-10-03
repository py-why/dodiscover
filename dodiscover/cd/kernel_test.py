from typing import Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from dodiscover.ci.kernel_test import compute_kernel
from dodiscover.typing import Column

from .base import BaseConditionalDiscrepancyTest


def _compute_W_matrix(K: NDArray, z, l0, l1):
    """Compute W matrices as done in KCD test.

    Parameters
    ----------
    K : NDArray of shape (n_samples, n_samples)
        The kernel matrix.
    z : NDArray of shape (n_samples)
        The indicator variable of 1's and 0's for which samples belong
        to which group.
    l0 : float
        The l2 regularization penalty applied to the inverse problem for the
        ``W_0`` matrix.
    l1 : float
        The l2 regularization penalty applied to the inverse problem for the
        ``W_1`` matrix.

    Returns
    -------
    W0 : NDArray

    W1 : NDArray

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

    K0 = K[first_mask, first_mask]
    K1 = K[second_mask, second_mask]

    # compute the number of samples in each
    n0 = int(np.sum(1 - z))
    n1 = int(np.sum(z))

    W0 = np.linalg.inv(K0 + l0 * np.identity(n0))
    W1 = np.linalg.inv(K1 + l1 * np.identity(n1))
    return W0, W1


def _estimate_propensity_scores(K, z, penalty=None, n_jobs=None, random_state=None):
    if penalty is None:
        penalty = _default_regularization(K)

    clf = LogisticRegression(
        penalty="l2",
        n_jobs=n_jobs,
        warm_start=True,
        solver="lbfgs",
        random_state=random_state,
        C=1 / (2 * penalty),
    )

    # fit and then obtain the probabilities of treatment
    # for each sample (i.e. the propensity scores)
    e_hat = clf.fit(K, z).predict_proba(K)[:, 1]

    return e_hat


def _default_regularization(K):
    n_samples = K.shape[0]
    svals = np.linalg.svd(K, compute_uv=False, hermitian=True)
    res = minimize_scalar(
        lambda reg: np.sum(svals**2 / (svals + reg) ** 2) / n_samples + reg,
        bounds=(0.0001, 1000),
        method="bounded",
    )
    return res.x


class KernelCDTest(BaseConditionalDiscrepancyTest):
    def __init__(
        self,
        distance_metric="euclidean",
        metric="rbf",
        l2=None,
        kwidth_x=None,
        kwidth_y=None,
        null_reps: int = 1000,
        n_jobs=None,
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

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        group_col: Column,
    ) -> Tuple[float, float]:
        x_cols = list(x_vars)
        y_cols = list(y_vars)

        group_ind = df[group_col].to_numpy()
        if set(np.unique(group_ind)) != {0, 1}:
            raise RuntimeError(f"Group indications in {group_col} column should be all 1 or 0.")

        # compute kernel
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
        self.propensity_penalty_ = _default_regularization(K)
        e_hat = _estimate_propensity_scores(
            K,
            group_ind,
            penalty=self.propensity_penalty_,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        # now compute null distribution
        null_dist = self.compute_null(
            e_hat, K, L, null_reps=self.null_reps, random_state=self.random_state
        )
        self.null_dist_ = null_dist

        # compute the pvalue
        pvalue = (1 + np.sum(null_dist >= stat)) / (1 + self.null_reps)
        return stat, pvalue

    def _statistic(self, K: NDArray, L: NDArray, group_ind: NDArray) -> float:
        n_samples = len(K)

        # compute W matrices from K
        W0, W1 = _compute_W_matrix(K, group_ind, l0=self.l2, l1=self.l2)

        # compute L kernels
        first_mask = np.array(1 - group_ind, dtype=bool)
        second_mask = np.array(group_ind, dtype=bool)
        L0 = L[first_mask, first_mask]
        L1 = L[second_mask, second_mask]
        L01 = L[first_mask, second_mask]

        # compute the final test statistic
        K0 = K[:, first_mask]
        K1 = K[:, second_mask]

        # compute the three terms in Lemma 4.4
        first_term = np.trace((K0 @ W0).T @ L0 @ (K0 @ W0))
        second_term = np.trace((K1 @ W1).T @ L01 @ (K0 @ W0))
        third_term = np.trace((K1 @ W1).T @ L1 @ (K1 @ W1))

        # compute final statistic
        stat = (first_term - 2 * second_term + third_term) / n_samples
        return stat

    def compute_null(self, e_hat, K, L, null_reps=1000, random_state=None):
        rng = np.random.default_rng(random_state)

        # compute the test statistic on the conditionally permuted
        # dataset, where each group label is resampled for each sample
        # according to its propensity score
        null_dist = Parallel(n_jobs=self.n_jobs)(
            [delayed(self._statistic)(K, L, rng.binomial(1, e_hat)) for _ in range(null_reps)]
        )
        return null_dist
