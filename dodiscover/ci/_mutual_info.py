import numpy as np
import scipy.stats
from scipy.special import digamma


def _estimate_entropy_c(x, y=None, n_nbrs=3, base=None, axis=0, random_state=None):
    """Compute continuous entropy using the K-L kNN estimator.

    The Kozachenko and Leonenko kNN estimator is used to estimate
    (conditional) entropy using a kNN approach :footcite:`Kozachenko1987sample`.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_dims_x)
        The X variable.
    y : array-like of shape (n_samples, n_dims_y), optional
        The Y variable, by default None
    n_nbrs : int, optional
        Number of neighbors to use in the kNN, by default 3.
    base : float, optional
        The logarithmic base, by default None, which corresponds to
        the natural logarithm for bits.
    axis : int, optional
        The axis along which to compute entropy, by default 0.
    random_state : int, optional
        The random seed.

    Notes
    -----
    For discrete variable entropy, use :func:`scipy.stats.entropy`.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_dims = x.shape

    # add noise along each dimension to prevent ties
    x = x + rng.random(size=(n_samples, n_dims)) * 1.0e-8

    if base is None:
        base = 2

    if y is not None:
        # repeat the analysis but with the joint and marginal
        xy = np.hstack((x, y))
        ent_xy = _estimate_entropy_c(
            xy, n_nbrs=n_nbrs, base=base, axis=axis, random_state=random_state
        )
        ent_y = _estimate_entropy_c(
            y, n_nbrs=n_nbrs, base=base, axis=axis, random_state=random_state
        )
        ent = ent_xy - ent_y
    else:
        # then build a tree
        nn_tree = scipy.stats.cKDTree(x)
        nn = [nn_tree.query(x_point, n_nbrs + 1, p="inf")[0][n_nbrs] for x_point in x]

        const = digamma(n_samples) - digamma(n_nbrs) + n_dims * np.log(2)
        ent = (const + n_dims * np.log(nn).mean()) / np.log(base)
    return ent
