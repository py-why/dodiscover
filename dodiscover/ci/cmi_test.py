from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import scipy.special
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler

from dodiscover.typing import Column

from .base import BaseConditionalIndependenceTest, CMIMixin


class CMITest(BaseConditionalIndependenceTest, CMIMixin):
    r"""Conditional mutual information independence test.

    Implements the conditional independence test using conditional
    mutual information proposed in :footcite:`Runge2018cmi`.

    Parameters
    ----------
    k : float, optional
        Number of nearest-neighbors for each sample point. If the number is
        smaller than 1, it is computed as a fraction of the number of
        samples, by default 0.2.
    transform : str, optional
        Transform the data by standardizing the data, by default 'rank', which converts
        data to ranks. Can be 'rank', 'uniform', 'standardize'.
    n_jobs : int, optional
        The number of CPUs to use, by default -1, which corresponds to
        using all CPUs available.
    n_shuffle_nbrs : int, optional
        Number of nearest-neighbors within the Z covariates for shuffling, by default 5.
    n_shuffle : int
        The number of times to shuffle the dataset to generate the null distribution.
        By default, 1000.
    random_seed : int, optional
        The random seed that is used to seed via ``np.random.defaultrng``.

    Notes
    -----
    Conditional mutual information (CMI) is defined as:

    .. math::

        I(X;Y|Z) = \iiint p(z) p(x,y|z)
        \log \frac{ p(x,y|z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    It can be seen that when :math:`X \perp Y | Z`, then CMI is equal to 0.
    Hence, CMI is a general measure for conditional dependence. The
    estimator for CMI proposed in :footcite:`Runge2018cmi` is a
    k-nearest-neighbor based estimator:

    .. math::

        \widehat{I}(X;Y|Z) = \psi (k) + \frac{1}{T} \sum_{t=1}^T
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

    random_state: np.random.Generator

    def __init__(
        self,
        k: float = 0.2,
        transform: str = "rank",
        n_jobs: int = -1,
        n_shuffle_nbrs: int = 5,
        n_shuffle: int = 100,
        random_seed: Optional[int] = None,
    ) -> None:
        self.k = k
        self.n_shuffle_nbrs = n_shuffle_nbrs
        self.transform = transform
        self.n_jobs = n_jobs
        self.n_shuffle = n_shuffle
        self.random_seed = random_seed
        self.random_state = np.random.default_rng(self.random_seed)

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set] = None,
    ) -> Tuple[float, float]:
        if z_covariates is None:
            z_covariates = set()

        self._check_test_input(df, x_vars, y_vars, z_covariates)

        # preprocess and transform the data; called here only once
        df = self._preprocess_data(df)

        # compute the estimate of the CMI
        val = self._compute_cmi(df, x_vars, y_vars, z_covariates)

        # compute the significance of the CMI value
        null_dist = self._estimate_null_dist(
            df,
            x_vars,
            y_vars,
            z_covariates,
            n_shuffle_nbrs=self.n_shuffle_nbrs,
            n_shuffle=self.n_shuffle,
        )

        # compute pvalue
        pvalue = (null_dist >= val).mean()

        self.stat_ = val
        self.pvalue_ = pvalue
        self.null_dist_ = null_dist
        return val, pvalue

    def _compute_cmi(self, df, x_vars, y_vars, z_covariates: Set):
        n_samples, _ = df.shape

        if self.k < 1:
            knn_here = max(1, int(self.k * n_samples))
        else:
            knn_here = max(1, int(self.k))

        # compute the K nearest neighbors in sub-spaces
        k_xz, k_yz, k_z = self._get_knn(df, x_vars, y_vars, z_covariates)

        # compute the final CMI value
        hxyz = scipy.special.digamma(knn_here)
        hxz = scipy.special.digamma(k_xz)
        hyz = scipy.special.digamma(k_yz)
        hz = scipy.special.digamma(k_z)
        val = hxyz - (hxz + hyz - hz).mean()
        return val

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        n_samples, n_dims = data.shape

        # make a copy of the data to prevent changing it
        data = data.copy()

        # add minor noise to make sure there are no ties
        random_noise = self.random_state.random((n_samples, n_dims))
        data += 1e-5 * random_noise @ data.std(axis=0).to_numpy().reshape(n_dims, 1)

        if self.transform == "standardize":
            # standardize with standard scaling
            data = data.astype(np.float64)
            scaler = StandardScaler()
            data[data.columns] = scaler.fit_transform(data[data.columns])
        elif self.transform == "uniform":
            data = self._trafo2uniform(data)
        elif self.transform == "rank":
            # rank transform each column
            data = data.rank(axis=0)
        return data

    def _get_knn(
        self, data: pd.DataFrame, x_vars: Set[Column], y_vars: Set[Column], z_covariates: Set
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Compute the nearest neighbor in the variable subspaces.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        x_var : Set[Column]
            The X variable column(s).
        y_var : Set[Column]
            The Y variable column(s).
        z_covariates : Set[Column]
            The Z variable column(s).

        Returns
        -------
        k_xz : np.ArrayLike of shape (n_samples,)
            Nearest neighbors in subspace of ``x_var`` and the
            ``z_covariates``.
        k_yz : np.ArrayLike of shape (n_samples,)
            Nearest neighbors in subspace of ``y_var`` and the
            ``z_covariates``.
        k_z : np.ArrayLike of shape (n_samples,)
            Nearest neighbors in subspace of ``z_covariates``.
        """
        n_samples, _ = data.shape
        if self.k < 1:
            knn = max(1, int(self.k * n_samples))
        else:
            knn = max(1, int(self.k))
        xz_cols = list(x_vars)
        for z_var in z_covariates:
            xz_cols.append(z_var)
        yz_cols = list(y_vars)
        for z_var in z_covariates:
            yz_cols.append(z_var)
        z_columns = list(z_covariates)
        columns = list(set(xz_cols).union(set(yz_cols)))
        data = data[columns]

        tree_xyz = scipy.spatial.KDTree(data.to_numpy())
        epsarray = tree_xyz.query(
            data.to_numpy(), k=[knn + 1], p=np.inf, eps=0.0, workers=self.n_jobs
        )[0][:, 0].astype(np.float64)

        # To search neighbors < eps
        epsarray = np.multiply(epsarray, 0.99999)

        # Find nearest neighbors in subspaces of X and Z
        xz = data[xz_cols].to_numpy()
        tree_xz = scipy.spatial.KDTree(xz)
        k_xz = tree_xz.query_ball_point(
            xz, r=epsarray, eps=0.0, p=np.inf, workers=self.n_jobs, return_length=True
        )

        # Find nearest neighbors in subspaces of Y and Z
        yz = data[yz_cols].to_numpy()
        tree_yz = scipy.spatial.KDTree(yz)
        k_yz = tree_yz.query_ball_point(
            yz, r=epsarray, eps=0.0, p=np.inf, workers=self.n_jobs, return_length=True
        )

        # Find nearest neighbors in subspaces of just the Z covariates
        if len(z_columns) > 0:
            z = data[z_columns].to_numpy()
            tree_z = scipy.spatial.KDTree(z)
            k_z = tree_z.query_ball_point(
                z, r=epsarray, eps=0.0, p=np.inf, workers=self.n_jobs, return_length=True
            )
        else:
            # Number of neighbors is n_samples when estimating just standard MI
            k_z = np.full(n_samples, n_samples, dtype=np.float64)

        return k_xz, k_yz, k_z

    def _trafo2uniform(self, df):
        """Transforms input array to uniform marginals.

        Assumes x.shape = (dim, T)

        Parameters
        ----------
        df : pandas.DataFrame
            The input data with (n_samples,) rows and (n_features,) columns.

        Returns
        -------
        u : array-like
            array with uniform marginals.
        """

        def trafo(xi):
            xisorted = np.sort(xi)
            yi = np.linspace(1.0 / len(xi), 1, len(xi))
            return np.interp(xi, xisorted, yi)

        # apply a uniform transformation for each feature
        for idx in range(len(df.columns)):
            marginalized_feature = trafo(df.iloc[:, idx].to_numpy().squeeze())
            df.iloc[:, idx] = marginalized_feature
        return df
