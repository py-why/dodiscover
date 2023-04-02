from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import logm
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances, pairwise_kernels


def von_neumann_divergence(A: ArrayLike, B: ArrayLike) -> float:
    """Compute Von Neumann divergence between two PSD matrices.

    Parameters
    ----------
    A : ArrayLike of shape (n_features, n_features)
        The first PSD matrix.
    B : ArrayLike of shape (n_features, n_features)
        The second PSD matrix.

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


def f_divergence_score(y_stat_q: ArrayLike, y_stat_p: ArrayLike) -> float:
    r"""Compute f-divergence upper bound on KL-divergence.

    See definition 4 in :footcite:`Mukherjee2020ccmi`, where
    the function is reversed to give an upper-bound for the
    sake of gradient descent.

    The f-divergence bound gives an upper bound on KL-divergence:

    .. math::
        D_{KL}(p || q) \le \sup_f E_{x \sim q}[exp(f(x) - 1)] - E_{x \sim p}[f(x)]

    Parameters
    ----------
    y_stat_q : ArrayLike of shape (n_samples_q,)
        Samples from the distribution Q, the variational class.
    y_stat_p : : ArrayLike of shape (n_samples_p,)
        Samples from the distribution P, the joint distribution.

    Returns
    -------
    f_div : float
        The f-divergence score.
    """
    f_div = np.mean(np.exp(y_stat_q - 1)) - np.mean(y_stat_p)
    return f_div


def kl_divergence_score(y_stat_q: ArrayLike, y_stat_p: ArrayLike, eps: float) -> float:
    r"""Compute f-divergence upper bound on KL-divergence.

    See definition 4 in :footcite:`Mukherjee2020ccmi`, where
    the function is reversed to give an upper-bound for the
    sake of gradient descent.

    The KL-divergence can be estimated with the following formula:

    .. math::
        \hat{D}_{KL}(p || q) = \frac{1}{n} \sum_{i=1}^n log L(Y_i^p) -
        log (\frac{1}{m} \sum_{j=1}^m L(Y_j^q))

    Parameters
    ----------
    y_stat_q : ArrayLike of shape (n_samples_q,)
        Samples from the distribution Q, the variational class.
        This corresponds to :math:`Y_j^q` samples.
    y_stat_p : : ArrayLike of shape (n_samples_p,)
        Samples from the distribution P, the joint distribution.
        This corresponds to :math:`Y_i^p` samples.

    Returns
    -------
    metric : float
        The KL-divergence score.
    """
    # compute point-wise likelihood ratio for each prediction
    p_l_ratio = (y_stat_p + eps) / np.abs(1 - y_stat_p - eps)
    q_l_ratio = (y_stat_q + eps) / np.abs(1 - y_stat_q - eps)

    # now compute KL-divergence estimate
    kldiv = np.mean(np.log(p_l_ratio)) - np.log(np.mean(q_l_ratio))
    return kldiv


def corrent_matrix(
    data: ArrayLike,
    metric: str = "rbf",
    kwidth: Optional[float] = None,
    distance_metric="euclidean",
    n_jobs: Optional[int] = None,
) -> ArrayLike:
    """Compute the centered correntropy of a matrix.

    Parameters
    ----------
    data : ArrayLike of shape (n_samples, n_features)
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
    data : ArrayLike of shape (n_features, n_features)
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
                data[:, idx][:, np.newaxis],
                data[:, jdx][:, np.newaxis],
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
    X: ArrayLike,
    method="scott",
    distance_metric: Optional[str] = None,
    n_jobs: Optional[int] = None,
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


def _default_regularization(K: ArrayLike) -> float:
    """Computes a default regularization for Kernel Logistic Regression.

    Parameters
    ----------
    K : ArrayLike of shape (n_samples, n_samples)
        The kernel matrix.

    Returns
    -------
    x : float
        The default l2 regularization term.
    """
    n_samples = K.shape[0]
    svals = np.linalg.svd(K, compute_uv=False, hermitian=True)
    res = minimize_scalar(
        lambda reg: np.sum(svals**2 / (svals + reg) ** 2) / n_samples + reg,
        bounds=(0.0001, 1000),
        method="bounded",
    )
    return res.x
