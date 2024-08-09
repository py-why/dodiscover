from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state


def nonlinear_additive_gaussian(
    model_type: str,
    n_samples: int = 1000,
    dims_x: int = 1,
    dims_y: int = 1,
    dims_z: int = 1,
    std: float = 0.5,
    freq: float = 1.0,
    cause_var: Optional[NDArray] = None,
    nonlinear_func: Callable = np.cos,
    random_state=None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate samples from a cosine nonlinear model with additive noise.

    The data comes in one of three forms:
    - :math:`X \\perp Y | Z`, where ``X <- Z -> Y``
    - all X, Y, Z are independent
    - ``X -> Y <- Z``, where :math:`X \\perp Z`, but :math:`X \\not\\perp Z | Y`

    Follows simulation setup of :footcite:`Sen2017model` for
    "post-nonlinear noise" model.

    Parameters
    ----------
    model_type : str
        'ci', 'dep', or 'ind' for conditional independence,
        dependent or completely independent.
    n_samples : int, optional
        Number of samples to generate, by default 1000
    dims_x : int, optional
        Dimensionality of X, by default 1.
    dims_y : int, optional
        Dimensionality of Y, by default 1.
    dims_z : int, optional
        Dimensionality of Z, by default 1.
    std : float, optional
        Noise amplitude, by default 0.5.
    freq : float, optional
        Frequency of the cosine signal, by default 1.0.
    random_state : random state, optional
        Random seed, by default None.

    Returns
    -------
    X : NDArray of shape (n_samples, dims_x)
        The X array.
    Y : NDArray of shape (n_samples, dims_y)
        The Y array.
    Z : NDArray of shape (n_samples, dims_z)
        The Z array.

    References
    ----------
    .. footbibliography::
    """
    rng = check_random_state(random_state)

    cov = np.eye(dims_z)
    mu = np.ones(dims_z)

    # generate (n_samples x dims_z) Z variable
    Z = rng.multivariate_normal(mu, cov, n_samples)

    # generate random weighting from Z to X
    # compute the column sums and normalize
    Azx = rng.rand(dims_z, dims_x)
    col_sum = np.linalg.norm(Azx, axis=0, keepdims=True)
    Azx = Azx / col_sum

    # generate random weighting from Z to Y
    Azy = rng.rand(dims_z, dims_y)
    col_sum = np.linalg.norm(Azy, axis=0, keepdims=True)
    Azy = Azy / col_sum

    # generate random weighting from X to Y
    Axy = rng.rand(dims_x, dims_y)
    col_sum = np.linalg.norm(Axy, axis=0, keepdims=True)
    Axy = Axy / col_sum

    # the sampled multivariate noises
    X_noise = rng.multivariate_normal(np.zeros(dims_x), np.eye(dims_x), n_samples)
    Y_noise = rng.multivariate_normal(np.zeros(dims_y), np.eye(dims_y), n_samples)

    if cause_var is None:
        cause_var = 0

    # compute nonlinear model
    if model_type == "ci":
        # X <- Z -> Y
        X = nonlinear_func(freq * (Z * Azx + std * X_noise + cause_var))
        Y = nonlinear_func(freq * (Z * Azy + std * Y_noise + cause_var))
    elif model_type == "ind":
        # X, Y, Z
        X = nonlinear_func(freq * (std * X_noise + cause_var))
        Y = nonlinear_func(freq * (std * Y_noise + cause_var))
    elif model_type == "dep":
        # X -> Y <- Z
        X = nonlinear_func(freq * (std * X_noise + cause_var))
        Y = nonlinear_func(freq * (2 * Axy * X + Z * Azy + std * Y_noise + cause_var))

    return X, Y, Z
