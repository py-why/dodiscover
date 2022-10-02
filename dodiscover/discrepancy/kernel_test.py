
from typing import Set, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dodiscover.typing import Column


def _compute_W_matrix(K: NDArray, z, l0, l1):
    """Compute W matrices as done in KCD test.

    Parameters
    ----------
    K : NDArray
        _description_
    z : _type_
        _description_

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

    W0 = np.linalg.inv(K0 + l0 * np.identity(n0))
    W1 = np.linalg.inv(K1 + l1 * np.identity(n1))
    return W0, W1

class KernelCDTest:

    def __init__(self, dist_func = None, l2=None, n_jobs=None, **dist_func_kwargs) -> None:
        self.l2 = l2
        self.n_jobs = n_jobs

        self.dist_func = dist_func
        self.dist_func_kwargs = dist_func_kwargs

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        x_cols = list(x_vars)
        y_cols = list(y_vars)
        z_cols = list(z_covariates)

        # compute kernel


