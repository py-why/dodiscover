from typing import Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.linalg import logm
from scipy.stats import gaussian_kde

from dodiscover.ci.kernel_test import compute_kernel
from dodiscover.typing import Column

from .base import BaseConditionalDiscrepancyTest


def von_neumann_divergence(A: NDArray, B: NDArray) -> float:
    """Compute Von Neumann divergence between two PSD matrices.

    Parameters
    ----------
    A : NDArray of shape (n, n)
        The first PSD matrix.
    B : NDArray of shape (n, n)
        The second PSD matrix

    Returns
    -------
    div : float
        The divergence value.

    Notes
    -----
    The Von Neumann divergence, or what is known as the Bregman divergence in
    :footcite:`Yu2020Bregman` is computed as follows with
    :math:`D(A || B) = Tr(A (log(A) - log(B)) - A + B)`.
    """
    div = np.trace(A.dot(logm(A) - logm(B)) - A + B)
    return div


def corrent_matrix(
    data: NDArray,
    metric: str = "rbf",
    kwidth: float = None,
    distance_metric="euclidean",
    n_jobs=None,
) -> NDArray:
    """Compute the centered correntropy of a matrix.

    Parameters
    ----------
    data : NDArray of shape (n_samples, n_features)
        The data.
    metric : str
        The kernel metric.
    kwidth : float
        The kernel width.
    distance_metric : str
        The distance metric to infer kernel width.
    n_jobs : int, optional
        The number of jobs to run computations in parallel, by default None.

    Returns
    -------
    data : NDArray of shape (n_features, n_features)
        A symmetric centered correntropy matrix of the data.

    Notes
    -----
    The estimator for the correntropy array is given by the formula
    :math:`1 / N \\sum_{i=1}^N k(x_i, y_i) - 1 / N**2 \\sum_{i=1}^N \\sum_{j=1}^N k(x_i, y_j)`.
    The first term is the estimate, and the second term is the bias, and together they form
    an unbiased estimate.
    """
    n_samples, n_features = data.shape
    corren_arr = np.zeros(shape=(n_features, n_features))

    # compute kernel between each feature, which is now (n_features, n_features) array
    for idx in range(n_features):
        for jdx in range(idx + 1):
            K, kwidth = compute_kernel(
                data[:, idx],
                data[:, jdx],
                metric=metric,
                distance_metric=distance_metric,
                kwidth=kwidth,
                centered=False,
                n_jobs=n_jobs,
            )

            # compute the bias due to finite-samples
            bias = np.sum(K) / n_samples**2

            # compute the sample centered correntropy
            corren = (1.0 / n_samples) * np.trace(K) - bias

            corren_arr[idx, jdx] = corren_arr[jdx, idx] = corren
    return corren_arr


def _estimate_kwidth(data, method="scott"):
    kde = gaussian_kde(data)

    if method == "scott":
        kwidth = kde.scotts_factor()
    elif method == "silverman":
        kwidth = kde.silverman_factor()
    return kwidth


class BregmanCDTest(BaseConditionalDiscrepancyTest):
    def __init__(
        self, kwidth: float = None, null_reps: int = 1000, n_jobs: int = None, random_state=None
    ) -> None:
        self.kwidth = kwidth
        self.null_reps = null_reps
        self.n_jobs = n_jobs
        self.random_state = random_state

    def test(
        self, df: pd.DataFrame, x_vars: Set[Column], y_vars: Set[Column], group_col: Column
    ) -> Tuple[float, float]:
        x_cols = list(x_vars)
        y_cols = list(y_vars)
        group_ind = df[group_col].to_numpy()
        if set(np.unique(group_ind)) != {0, 1}:
            raise RuntimeError(f"Group indications in {group_col} column should be all 1 or 0.")

        # get the X and Y dataset
        X = df[x_cols].to_numpy()
        Y = df[y_cols].to_numpy()

        # We are interested in testing: P_1(y|x) = P_2(y|x)
        # compute the conditional divergence, which is symmetric by construction
        # 1/2 * (D(p_1(y|x) || p_2(y|x)) + D(p_2(y|x) || p_1(y|x)))
        conditional_div = self._statistic(X, Y, group_ind)

        # now compute null distribution
        null_dist = self.compute_null(
            X, Y, null_reps=self.null_reps, random_state=self.random_state
        )
        self.null_dist_ = null_dist

        # compute pvalue
        pvalue = (1.0 + np.sum(null_dist >= conditional_div)) / (1 + self.null_reps)
        return conditional_div, pvalue

    def _statistic(self, X: NDArray, Y: NDArray, group_ind: NDArray) -> float:
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
        Cx1 = corrent_matrix(X1, kwidth=self.kwidth)
        Cx2 = corrent_matrix(X2, kwidth=self.kwidth)

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

    def compute_null(self, X: NDArray, Y: NDArray, null_reps: int = 1000, random_state: int = None):
        rng = np.random.default_rng(random_state)

        p = 0.5
        n_samps = X.shape[0]

        # compute the test statistic on the conditionally permuted
        # dataset, where each group label is resampled for each sample
        # according to its propensity score
        null_dist = Parallel(n_jobs=self.n_jobs)(
            [
                delayed(self._statistic)(X, Y, rng.binomial(1, p, size=n_samps))
                for _ in range(null_reps)
            ]
        )
        return null_dist
