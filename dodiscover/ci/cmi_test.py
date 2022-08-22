from random import random
from typing import Any, Set, Tuple, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import scipy
import scipy.special
from scipy.stats import norm
import scipy.spatial
from warnings import warn
from .base import BaseConditionalIndependenceTest


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
    n_shuffle_nghbrs : int, optional
        Number of nearest-neighbors within the Z covariates for shuffling, by default 5.
    transform : str, optional
        Transform the data by standardizing the data, by default 'rank', which converts
        data to ranks.
    n_shuffle : int
        The number of samples to shuffle for a significance test.
    n_jobs : int, optional
        The number of CPUs to use, by default -1, which corrresponds to
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
    function.

    The estimator implemented here assumes the data is continuous.

    References
    ----------
    .. footbibliography::
    """
    def __init__(self, k: float=0.2, n_shuffle_nghbrs: int=5, transform: str='rank', n_jobs: int=-1,
        n_shuffle:int=1000, random_state=None) -> None:
        self.k=k
        self.n_shuffle_nghbrs = n_shuffle_nghbrs
        self.transform = transform
        self.n_jobs = n_jobs
        self.n_shuffle = n_shuffle

        if random_state is None:
            random_state = np.random.BitGenerator()
        self.random_state=random_state

    def test(self, df: pd.DataFrame, x_var, y_var, z_covariates = None) -> Tuple[float, float]:
        n_samples, _ = df.shape

        if self.k < 1:
            knn_here = max(1, int(self.k*n_samples))
        else:
            knn_here = max(1, int(self.k))
        
        # preprocess and transform the data
        df = self._preprocess_data(df)

        # compute the K nearest neighbors in sub-spaces
        k_xz, k_yz, k_z = self._get_knn(df, x_var, y_var, z_covariates)

        # compute the final CMI value
        val = scipy.special.digamma(knn_here) - (scipy.special.digamma(k_xz) +
                                           scipy.special.digamma(k_yz) -
                                           scipy.special.digamma(k_z)).mean()

        # compute the significance of the CMI value
        null_dist = self._estimate_null_dist(df, x_var, y_var, z_covariates, val)
        
        # compute pvalue
        pvalue = (null_dist >= val).mean()

        self.stat_ = val
        self.pvalue_ = pvalue
        self.null_dist_ = null_dist

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        n_samples, n_dims = data.shape
        
        # make a copy of the data to prevent changing it
        data = data.copy()

        # add minor noise to make sure there are no ties
        random_noise = self.random_state.random((n_samples, n_dims))
        data += 1E-6 * data.std(axis=0).reshape(n_dims, 1) * random_noise
        
        if self.transform == 'standardize':
            # standardize with standard scaling
            data = data.astype(np.float64)
            data -= data.mean(axis=0).reshape(n_dims, 1)
            std = data.std(axis=0)
            for i in range(n_dims):
                if std[i] != 0.:
                    data[i] /= std[i]

            if np.any(std == 0.):
                warn("Possibly constant data!")
        elif self.transform == 'uniform':
            data = self._trafo2uniform(data)
        elif self.transform == 'ranks':
            data = data.argsort(axis=0).argsort(axis=0).astype(np.float64)

        return data

    def _get_knn(self, data: pd.DataFrame, x_var, y_var, z_covariates) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute the nearest neighbor in the variable subspaces.

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

        tree_xyz = scipy.spatial.cKDTree(data.to_numpy())
        epsarray = tree_xyz.query(data.to_numpy(), k=[self.k+1], p=np.inf,
                                  eps=0., workers=self.n_jobs)[0][:, 0].astype(np.float64)

        # To search neighbors < eps
        epsarray = np.multiply(epsarray, 0.99999)

        # Subsample indices
        x_indices = list(data.columns).index(x_var)
        y_indices = list(data.columns).index(y_var)
        z_indices = [list(data.columns).index(z_var) for z_var in z_covariates]

        # Find nearest neighbors in subspaces of X and Z
        xz = data[:, np.concatenate((x_indices, z_indices))]
        tree_xz = scipy.spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(xz, r=epsarray, eps=0., p=np.inf, workers=self.workers, return_length=True)

        # Find nearest neighbors in subspaces of Y and Z
        yz = data[:, np.concatenate((y_indices, z_indices))]
        tree_yz = scipy.spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(yz, r=epsarray, eps=0., p=np.inf, workers=self.workers, return_length=True)

        # Find nearest neighbors in subspaces of just the Z covariates
        if len(z_indices) > 0:
            z = data[:, z_indices]
            tree_z = scipy.spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(z, r=epsarray, eps=0., p=np.inf, workers=self.workers, return_length=True)
        else:
            # Number of neighbors is n_samples when estimating just standard MI
            k_z = np.full(n_samples, n_samples, dtype=np.float64)

        return k_xz, k_yz, k_z

    def _estimate_null_dist(self, data: pd.DataFrame, x_var, y_var, z_covariates, value: float) -> float:
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
        value : float
            The estimated CMI test statistic.
        
        Returns
        -------
        pvalue : float
            The pvalue.
        """
        n_samples, n_dims = data.shape

        x_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_covariates) > 0 and self.n_shuffle_nghbrs < n_samples:
            # Get nearest neighbors around each sample point in Z
            z_array = data[:, z_covariates].to_numpy()
            tree_xyz = scipy.spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array,
                                       k=self.n_shuffle_nghbrs,
                                       p=np.inf,
                                       eps=0.)[1].astype(np.int32)

            null_dist = np.zeros(self.n_shuffle)
            for sam in range(self.n_shuffle):

                # Generate random order in which to go through indices loop in
                # next step
                order = self.random_state.permutation(T).astype(np.int32)

                # Shuffle neighbor indices for each sample index
                for i in range(len(neighbors)):
                    self.random_state.shuffle(neighbors[i])
                # neighbors = self.random_state.permuted(neighbors, axis=1)
                
                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = self.get_restricted_permutation(
                        T=n_samples,
                        n_shuffle_nghbrs=self.n_shuffle_nghbrs,
                        neighbors=neighbors,
                        order=order)

                array_shuffled = np.copy(data)
                for i in x_indices:
                    array_shuffled[i] = data[i, restricted_permutation]

                null_dist[sam] = self.get_dependence_measure(array_shuffled,
                                                             data)

        else:
            null_dist = \
                    self._get_shuffle_dist(data, xyz,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        return null_dist
        # if return_null_dist:
        #     # Sort
        #     null_dist.sort()