from typing import Optional, Set, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import scipy.special
import sklearn.utils
from numpy.typing import NDArray

from .base import BaseConditionalIndependenceTest
from .utils import _restricted_permutation


class CMITest(BaseConditionalIndependenceTest):
    """Conditional mutual information independence test.

    Implements the conditional independence test using conditional
    mutual information proposed in :footcite:`Runge2018cmi`.

    Parameters
    ----------
    k : float, optional
        Number of nearest-neighbors for each sample point. If the number is
        smaller than 1, it is computed as a fraction of the number of
        samples, by default 0.2.
    n_shuffle_nbrs : int, optional
        Number of nearest-neighbors within the Z covariates for shuffling, by default 5.
    transform : str, optional
        Transform the data by standardizing the data, by default 'rank', which converts
        data to ranks.
    n_shuffle : int
        The number of samples to shuffle for a significance test.
    n_jobs : int, optional
        The number of CPUs to use, by default -1, which corresponds to
        using all CPUs available.

    Notes
    -----
    Conditional mutual information (CMI) is defined as:

    .. math:: I(X;Y|Z) = \iiint p(z) p(x,y|z) \log
        \frac{ p(x,y|z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    It can be seen that when :math:`X \perp Y | Z`, then CMI is equal to 0.
    Hence, CMI is a general measure for conditional dependence. The
    estimator for CMI proposed in :footcite:`Runge2018cmi` is a
    k-nearest-neighbor based estimator:

    .. math:: \widehat{I}(X;Y|Z) = \psi (k) + \frac{1}{T} \sum_{t=1}^T
            (\psi(k_{Z,t}) - \psi(k_{XZ,t}) - \psi(k_{YZ,t}))

    where :math:`\psi` is the Digamma (i.e. see `scipy.special.digamma`)
    function. :math:`k` determines the
    size of hyper-cubes around each (high-dimensional) sample point. Then
    :math:`k_{Z,},k_{XZ},k_{YZ}` are the numbers of neighbors in the respective
    subspaces. :math:`k` can be viewed as a density smoothing parameter (although
    it is data-adaptive unlike fixed-bandwidth estimators). For large :math:`k`, the
    underlying dependencies are more smoothed and CMI has a larger bias,
    but lower variance, which is more important for significance testing. Note
    that the estimated CMI values can be slightly negative while CMI is a non-
    negative quantity.

    The estimator implemented here assumes the data is continuous.

    References
    ----------
    .. footbibliography::
    """

    random_state: np.random.BitGenerator

    def __init__(
        self,
        k: float = 0.2,
        n_shuffle_nbrs: int = 5,
        transform: str = "rank",
        n_jobs: int = -1,
        n_shuffle: int = 1000,
        random_state=None,
    ) -> None:
        self.k = k
        self.n_shuffle_nbrs = n_shuffle_nbrs
        self.transform = transform
        self.n_jobs = n_jobs
        self.n_shuffle = n_shuffle

        if random_state is None:
            random_state = np.random.RandomState(seed=random_state)
        self.random_state = random_state

    def test(
        self, df: pd.DataFrame, x_var, y_var, z_covariates: Optional[Set] = None
    ) -> Tuple[float, float]:
        if z_covariates is None:
            z_covariates = set()
        # compute the estimate of the CMI
        val = self._compute_cmi(df, x_var, y_var, z_covariates)

        # compute the significance of the CMI value
        null_dist = self._estimate_null_dist(df, x_var, y_var, z_covariates)

        # compute pvalue
        pvalue = (null_dist >= val).mean()

        self.stat_ = val
        self.pvalue_ = pvalue
        self.null_dist_ = null_dist
        return val, pvalue

    def _compute_cmi(self, df, x_var, y_var, z_covariates: Set):
        n_samples, _ = df.shape

        # preprocess and transform the data
        df = self._preprocess_data(df)

        if self.k < 1:
            knn_here = max(1, int(self.k * n_samples))
        else:
            knn_here = max(1, int(self.k))

        # compute the K nearest neighbors in sub-spaces
        k_xz, k_yz, k_z = self._get_knn(df, x_var, y_var, z_covariates)

        # compute the final CMI value
        val = (
            scipy.special.digamma(knn_here)
            - (
                scipy.special.digamma(k_xz)
                + scipy.special.digamma(k_yz)
                - scipy.special.digamma(k_z)
            ).mean()
        )
        return val

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        n_samples, n_dims = data.shape

        # make a copy of the data to prevent changing it
        data = data.copy()

        # add minor noise to make sure there are no ties
        random_noise = self.random_state.random((n_samples, n_dims))
        data += 1e-6 * random_noise @ data.std(axis=0).to_numpy().reshape(n_dims, 1)

        if self.transform == "standardize":
            # standardize with standard scaling
            data = data.astype(np.float64)
            data -= data.mean(axis=0).reshape(n_dims, 1)
            std = data.std(axis=0)
            for i in range(n_dims):
                if std[i] != 0.0:
                    data[i] /= std[i]

            if np.any(std == 0.0):
                warn("Possibly constant data!")
        elif self.transform == "uniform":
            data = self._trafo2uniform(data)
        elif self.transform == "rank":
            # rank transform each column
            data = data.rank(axis=0)

        return data

    def _get_knn(
        self, data: pd.DataFrame, x_var, y_var, z_covariates: Set
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute the nearest neighbor in the variable subspaces.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        x_var : column
            The X variable column.
        y_var : column
            The Y variable column.
        z_covariates : set of column
            The Z variable column(s).

        Returns
        -------
        k_xz : np.ndarray of shape (n_samples,)
            Nearest neighbors in subspace of ``x_var`` and the
            ``z_covariates``.
        k_yz : np.ndarray of shape (n_samples,)
            Nearest neighbors in subspace of ``y_var`` and the
            ``z_covariates``.
        k_z : np.ndarray of shape (n_samples,)
            Nearest neighbors in subspace of ``z_covariates``.
        """
        n_samples, _ = data.shape
        if self.k < 1:
            knn = max(1, int(self.k*n_samples))
        else:
            knn = max(1, int(self.k))

        tree_xyz = scipy.spatial.cKDTree(data.to_numpy())
        epsarray = tree_xyz.query(
            data.to_numpy(), k=[knn + 1], p=np.inf, eps=0.0, workers=self.n_jobs
        )[0][:, 0].astype(np.float64)

        # To search neighbors < eps
        epsarray = np.multiply(epsarray, 0.99999)

        # Find nearest neighbors in subspaces of X and Z
        xz_cols = [x_var]
        for z_var in z_covariates:
            xz_cols.append(z_var)
        xz = data[xz_cols]
        tree_xz = scipy.spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(
            xz, r=epsarray, eps=0.0, p=np.inf, workers=self.n_jobs, return_length=True
        )

        # Find nearest neighbors in subspaces of Y and Z
        yz_cols = [y_var]
        for z_var in z_covariates:
            yz_cols.append(z_var)
        yz = data[yz_cols]
        tree_yz = scipy.spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(
            yz, r=epsarray, eps=0.0, p=np.inf, workers=self.n_jobs, return_length=True
        )

        # Find nearest neighbors in subspaces of just the Z covariates
        if len(z_covariates) > 0:
            z = data[z_covariates]
            tree_z = scipy.spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(
                z, r=epsarray, eps=0.0, p=np.inf, workers=self.n_jobs, return_length=True
            )
        else:
            # Number of neighbors is n_samples when estimating just standard MI
            k_z = np.full(n_samples, n_samples, dtype=np.float64)

        return k_xz, k_yz, k_z

    def _estimate_null_dist(self, data: pd.DataFrame, x_var, y_var, z_covariates: Set) -> float:
        """Compute pvalue by performing a nearest-neighbor shuffle test.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        x_var : column
            The X variable column.
        y_var : column
            The Y variable column.
        z_covariates : column
            The Z variable column(s).

        Returns
        -------
        pvalue : float
            The pvalue.
        """
        n_samples, _ = data.shape

        if len(z_covariates) > 0 and self.n_shuffle_nbrs < n_samples:
            # Get nearest neighbors around each sample point in Z subspace
            z_array = data[z_covariates].to_numpy()
            tree_xyz = scipy.spatial.cKDTree(z_array)
            nbrs = tree_xyz.query(z_array, k=self.n_shuffle_nbrs, p=np.inf, eps=0.0)[1].astype(
                np.int32
            )

            null_dist = np.zeros(self.n_shuffle)

            # create a copy of the data to store the shuffled array
            data_copy = data.copy()

            # we will now compute a shuffled distribution, where X_i is replaced
            # by X_j only if the corresponding Z_i and Z_j are "nearby", computed
            # using the spatial tree query
            for idx in range(self.n_shuffle):
                # Shuffle neighbor indices for each sample index
                for i in range(len(nbrs)):
                    self.random_state.shuffle(nbrs[i])

                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = _restricted_permutation(
                    nbrs, self.n_shuffle_nbrs, n_samples=n_samples, random_state=self.random_state
                )

                # update the X variable column
                data_copy.loc[:, x_var] = data.loc[restricted_permutation, x_var].to_numpy()

                # compute the CMI on the shuffled array
                null_dist[idx] = self._compute_cmi(data_copy, x_var, y_var, z_covariates)
        else:
            null_dist = self._compute_shuffle_dist(
                data, x_var, y_var, z_covariates, n_shuffle=self.n_shuffle
            )

        return null_dist

    def _compute_shuffle_dist(
        self, data: pd.DataFrame, x_var, y_var, z_covariates: Set, n_shuffle: int
    ) -> NDArray:
        """Compute a shuffled distribution of the test statistic."""
        data_copy = data.copy()

        # initialize the shuffle distribution
        shuffle_dist = np.zeros((n_shuffle,))
        for idx in range(n_shuffle):
            # compute a shuffled version of the data
            shuffled_x = sklearn.utils.shuffle(data[x_var], random_state=self.random_state)

            # compute now the test statistic on the shuffle data
            data_copy[x_var] = shuffled_x
            shuffle_dist[idx] = self._compute_cmi(data_copy, x_var, y_var, z_covariates)

        return shuffle_dist
