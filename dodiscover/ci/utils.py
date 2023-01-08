from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, pairwise_kernels


def compute_kernel(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    metric: str = "rbf",
    distance_metric: str = "euclidean",
    kwidth: Optional[float] = None,
    centered: bool = True,
    n_jobs: Optional[int] = None,
) -> Tuple[ArrayLike, float]:
    """Compute a kernel matrix and corresponding width.

    Also optionally estimates the kernel width parameter.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples_X, n_features_X)
        The X array.
    Y : ArrayLike of shape (n_samples_Y, n_features_Y), optional
        The Y array, by default None.
    metric : str, optional
        The metric to compute the kernel function, by default 'rbf'.
        Can be any string as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`. Note 'rbf'
        and 'gaussian' are the same metric.
    distance_metric : str, optional
        The distance metric to compute distances among samples within
        each data matrix, by default 'euclidean'. Can be any valid string
        as defined in :func:`sklearn.metrics.pairwise_distances`.
    kwidth : float, optional
        The kernel width, by default None, which will then be estimated as the
        median L2 distance between the X features.
    centered : bool, optional
        Whether to center the kernel matrix or not, by default True.
    n_jobs : int, optional
        The number of jobs to run computations in parallel, by default None.

    Returns
    -------
    kernel : ArrayLike of shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
        The kernel matrix.
    med : float
        The estimated kernel width.
    """
    # if the width of the kernel is not set, then use the median trick to set the
    # kernel width based on the data X
    if kwidth is None:
        med = _estimate_kwidth(X, method="median", distance_metric=distance_metric, n_jobs=n_jobs)
    else:
        med = kwidth

    extra_kwargs = dict()

    if metric == "rbf":
        # compute the normalization factor of the width of the Gaussian kernel
        gamma = 1.0 / (2 * (med**2))
        extra_kwargs["gamma"] = gamma
    elif metric == "polynomial":
        degree = 2
        extra_kwargs["degree"] = degree

    # compute the potentially pairwise kernel
    kernel = pairwise_kernels(X, Y=Y, metric=metric, n_jobs=n_jobs, **extra_kwargs)

    if centered:
        kernel = _center_kernel(kernel)
    return kernel, med


def _estimate_kwidth(
    X: ArrayLike, method="scott", distance_metric: str = None, n_jobs: int = None
) -> float:
    """Estimate kernel width.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        The data.
    method : str, optional
        Method to use, by default "scott".
    distance_metric : str, optional
        The distance metric to compute distances among samples within
        each data matrix, by default 'euclidean'. Can be any valid string
        as defined in :func:`sklearn.metrics.pairwise_distances`.
    n_jobs : int, optional
        The number of jobs to run computations in parallel, by default None.

    Returns
    -------
    kwidth : float
        The estimated kernel width for X.
    """

    if method == "scott":
        kde = gaussian_kde(X)
        kwidth = kde.scotts_factor()
    elif method == "silverman":
        kde = gaussian_kde(X)
        kwidth = kde.silverman_factor()
    elif method == "median":
        # Note: sigma = 1 / np.sqrt(kwidth)
        # compute N x N pairwise distance matrix
        dists = pairwise_distances(X, metric=distance_metric, n_jobs=n_jobs)

        # compute median of off diagonal elements
        med = np.median(dists[dists > 0])

        # prevents division by zero when used on label vectors
        kwidth = med if med else 1
    return kwidth


def _center_kernel(K: ArrayLike):
    """Centers a kernel matrix.

    Applies a transformation H * K * H, where H is a diagonal matrix with 1/n along
    the diagonal.

    Parameters
    ----------
    K : ArrayLike of shape (n_features, n_features)
        The kernel matrix.

    Returns
    -------
    K : ArrayLike of shape (n_features, n_features)
        The centered kernel matrix.
    """
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    return H.dot(K).dot(H)


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
