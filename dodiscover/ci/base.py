from abc import ABCMeta, abstractmethod
from typing import Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.utils
from numpy.typing import ArrayLike

from dodiscover.typing import Column

from .monte_carlo import generate_knn_in_subspace, restricted_nbr_permutation


class BaseConditionalIndependenceTest(metaclass=ABCMeta):
    """Abstract class for any conditional independence test.

    All CI tests are used in constraint-based causal discovery algorithms. This
    class interface is expected to be very lightweight to enable anyone to convert
    a function for CI testing into a class, which has a specific API.
    """

    # default CI tests do not allow multivariate input
    _allow_multivariate_input: bool = False

    def _check_test_input(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]],
    ):
        if any(col not in df.columns for col in x_vars):
            raise ValueError(f"The x variables {x_vars} are not all in the DataFrame.")
        if any(col not in df.columns for col in y_vars):
            raise ValueError(
                f"The y variables {y_vars} are not all in the DataFrame: {df.columns}."
            )
        if z_covariates is not None and any(col not in df.columns for col in z_covariates):
            raise ValueError(
                f"The z conditioning set variables {z_covariates} are not all in the "
                f"DataFrame with {df.columns}."
            )

        if not self._allow_multivariate_input and (len(x_vars) > 1 or len(y_vars) > 1):
            raise RuntimeError(f"{self.__class__} does not support multivariate input for X and Y.")

    @abstractmethod
    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        """Abstract method for all conditional independence tests.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        x_vars : Set of column
            A column in ``df``.
        y_vars : Set of column
            A column in ``df``.
        z_covariates : Set, optional
            A set of columns in ``df``, by default None. If None, then
            the test should run a standard independence test.

        Returns
        -------
        Tuple[float, float]
            Test statistic and pvalue.
        """
        pass


class ClassifierCIMixin:
    random_state: np.random.Generator
    test_size: Union[float, int]

    def generate_train_test_data(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
        k: int = 1,
    ) -> Tuple:
        """Generate a training and testing dataset for CCIT.

        This takes a conditional independence problem given a dataset
        and converts it to a binary classification problem.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        x_vars : Set of column
            A column in ``df``.
        y_vars : Set of column
            A column in ``df``.
        z_covariates : Set, optional
            A set of columns in ``df``, by default None. If None, then
            the test should run a standard independence test.
        k : int
            The K nearest-neighbors in subspaces for the conditional permutation
            step to generate distribution with conditional independence. By
            default, 1.

        Returns
        -------
        X_train, Y_train, X_test, Y_test : Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
            The X_train, y_train, X_test, y_test to be used in
            binary classification, where each dataset comprises
            of samples from the joint and conditionally independent
            distributions. ``y_train`` and ``y_test`` are comprised of
            1's and 0's only. Indices with value 1 indicate the original
            joint distribution, and indices with value 0 indicate
            the shuffled distribution.
        """
        if z_covariates is None:
            z_covariates = set()

        x_arr = df[list(x_vars)].to_numpy()
        y_arr = df[list(y_vars)].to_numpy()

        n_samples_x, _ = x_arr.shape
        test_size = self.test_size
        if test_size <= 1.0:
            test_size = int(test_size * n_samples_x)
        train_size = int(n_samples_x - test_size)

        # now slice the arrays to produce a training and testing dataset
        x_arr_train = x_arr[:train_size, :]
        y_arr_train = y_arr[:train_size, :]
        x_arr_test = x_arr[train_size:, :]
        y_arr_test = y_arr[train_size:, :]

        if len(z_covariates) == 0:
            n_samples_y, _ = y_arr.shape
            if n_samples_y != n_samples_x:
                raise ValueError(
                    f"There is unequal number of samples in x and y array: "
                    f"{n_samples_x} and {n_samples_y}."
                )

            # generate the training dataset
            X_ind, y_ind, X_joint, y_joint = self._unconditional_shuffle(x_arr_train, y_arr_train)
            X_train = np.vstack((X_ind, X_joint))
            Y_train = np.vstack((y_ind, y_joint))

            # generate the testing dataset
            X_ind, y_ind, X_joint, y_joint = self._unconditional_shuffle(x_arr_test, y_arr_test)
            X_test = np.vstack((X_ind, X_joint))
            Y_test = np.vstack((y_ind, y_joint))
        else:
            z_arr = df[list(z_covariates)].to_numpy()
            z_arr_train = z_arr[:train_size, :]
            z_arr_test = z_arr[train_size:, :]

            n_samples_x, _ = x_arr.shape
            n_samples_y, _ = y_arr.shape
            n_samples_z, _ = z_arr.shape
            if n_samples_y != n_samples_x or n_samples_z != n_samples_x:
                raise ValueError(
                    f"There is unequal number of samples in x, y and z array: "
                    f"{n_samples_x}, {n_samples_y}, {n_samples_z}."
                )

            # generate the training dataset
            X_ind, y_ind, X_joint, y_joint = self._conditional_shuffle(
                x_arr_train, y_arr_train, z_arr_train, k=k
            )
            X_train = np.vstack((X_ind, X_joint))
            Y_train = np.vstack((y_ind, y_joint))

            # generate the testing dataset
            X_ind, y_ind, X_joint, y_joint = self._conditional_shuffle(
                x_arr_test, y_arr_test, z_arr_test, k=k
            )
            X_test = np.vstack((X_ind, X_joint))
            Y_test = np.vstack((y_ind, y_joint))

        return X_train, Y_train, X_test, Y_test

    def _unconditional_shuffle(
        self, x_arr: ArrayLike, y_arr: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        r"""Generate samples to emulate X independent of Y.

        Based on input data ``(X, Y)``, partitions the dataset into halves, where
        one-half represents the joint distribution of ``P(X, Y)`` and the other
        represents the independent distribution of ``P(X)P(Y)``.

        Parameters
        ----------
        x_arr : ArrayLike of shape (n_samples, n_dims_x)
            The input X variable data.
        y_arr : ArrayLike of shape (n_samples, n_dims_y)
            The input Y variable data.

        Returns
        -------
        X_ind, y_ind, X_joint, y_joint : Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
            The X_ind, y_ind, X_joint, y_joint to be used in
            binary classification, where ``X_ind`` is the data features
            that are from the generated independent data and ``X_joint``
            is the data features from the original jointly distributed data.
            ``y_ind`` and ``y_joint`` correspond to the class labels 0 and 1
            respectively.

        Notes
        -----
        Shuffles the Y samples, such that if there is any dependence among X and Y,
        they are broken and the resulting samples emulate those which came from
        the distribution with :math:`X \\perp Y`. This essentially takes input
        data and generates a null distribution.
        """
        n_samples_x, _ = x_arr.shape

        # number of samples we will generate from the independent distribution
        n_ind_samples = n_samples_x // 2
        n_joint_samples = n_samples_x - n_ind_samples

        # generate half of the indices at random to generate the independent dataset
        rand_indices = self.random_state.choice(n_samples_x, size=n_ind_samples, replace=False)
        mask = np.ones((n_samples_x,), dtype=bool)
        mask[rand_indices] = False

        # subset the data into halves that maintain the joint distribution
        # and half that we will permute to create independence
        x_arr_ind = x_arr[rand_indices, :]
        x_arr_joint = x_arr[mask, :]
        y_arr_ind = y_arr[rand_indices, :]
        y_arr_joint = y_arr[mask, :]

        # now create independence in half of the data, while keeping
        # the joint distribution in the other half of the data
        indices = np.arange(n_ind_samples).astype(int)
        self.random_state.shuffle(indices)
        y_arr_ind = y_arr_ind[indices, :]

        # joint distributed data and independent data with their
        # corresponding class labels
        X_ind = np.hstack([x_arr_ind, y_arr_ind])
        y_ind = np.zeros((n_ind_samples, 1))
        X_joint = np.hstack([x_arr_joint, y_arr_joint])
        y_joint = np.ones((n_joint_samples, 1))

        return X_ind, y_ind, X_joint, y_joint

    def _conditional_shuffle(
        self,
        x_arr: ArrayLike,
        y_arr: ArrayLike,
        z_arr: ArrayLike,
        method="knn",
        k: int = 1,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Generate dataset for CI testing as binary classification problem.

        Based on input data ``(X, Y, Z)``, partitions the dataset into halves, where
        one-half represents the joint distribution of ``P(X, Y, Z)`` and the other
        represents the conditionally independent distribution of ``P(X | Z)P(Y)``.
        This is done by a nearest-neighbor bootstrap approach.

        Parameters
        ----------
        x_arr : ArrayLike of shape (n_samples, n_dims_x)
            The input X variable data.
        y_arr : ArrayLike of shape (n_samples, n_dims_y)
            The input Y variable data.
        z_arr : ArrayLike of shape (n_samples, n_dims_x)
            The input Z variable data.
        method : str, optional
            Method to use, by default 'knn'. Can be ('knn', 'kdtree').
        k : int, optional
            Number of nearest neighbors to swap, by default 1.

        Returns
        -------
        X_ind, y_ind, X_joint, y_joint : Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
            The X_ind, y_ind, X_joint, y_joint to be used in
            binary classification, where ``X_ind`` is the data features
            that are from the generated independent data and ``X_joint``
            is the data features from the original jointly distributed data.
            ``y_ind`` and ``y_joint`` correspond to the class labels 0 and 1
            respectively.

        Notes
        -----
        This algorithm implements a nearest-neighbor bootstrap approach for generating
        samples from the null hypothesis where :math:`X \\perp Y | Z`.
        """
        # use a method to generate k-nearest-neighbors in subspace of Z
        indices = generate_knn_in_subspace(z_arr, method=method, k=k)

        # Get the index of the sample kth closest to the 'idx' sample of Z
        # Then we will swap those samples in the Y variable
        y_prime = y_arr.copy()[indices[:, k], :]

        # construct the joint and conditionally independent distributions
        # and their corresponding class labels
        X_joint = np.hstack([x_arr, y_arr, z_arr])
        X_ind = np.hstack([x_arr, y_prime, z_arr])
        y_joint = np.ones((len(X_joint), 1))
        y_ind = np.zeros((len(X_ind), 1))

        return X_ind, y_ind, X_joint, y_joint


class CMIMixin:
    random_state: np.random.Generator
    random_seed: Optional[int]
    n_jobs: Optional[int]

    def _estimate_null_dist(
        self,
        data: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Set,
        n_shuffle_nbrs: int,
        n_shuffle: int,
    ) -> float:
        """Compute pvalue by performing a nearest-neighbor shuffle test.

        XXX: improve with parallelization with joblib

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        x_vars : Set[Column]
            The X variable column(s).
        y_var : Set[Column]
            The Y variable column(s).
        z_covariates : Set[Column]
            The Z variable column(s).
        n_shuffle_nbrs : int
            The number of nearest-neighbors in feature space to shuffle.
        n_shuffle : int
            The number of times to generate the shuffled distribution.

        Returns
        -------
        pvalue : float
            The pvalue.
        """
        n_samples, _ = data.shape

        x_cols = list(x_vars)
        if len(z_covariates) > 0 and n_shuffle_nbrs < n_samples:
            null_dist = np.zeros(n_shuffle)

            # create a copy of the data to store the shuffled array
            data_copy = data.copy()

            # Get nearest neighbors around each sample point in Z subspace
            z_array = data[list(z_covariates)].to_numpy()
            nbrs = generate_knn_in_subspace(
                z_array, method="kdtree", k=n_shuffle_nbrs, n_jobs=self.n_jobs
            )

            # we will now compute a shuffled distribution, where X_i is replaced
            # by X_j only if the corresponding Z_i and Z_j are "nearby", computed
            # using the spatial tree query
            for idx in range(n_shuffle):
                # Shuffle neighbor indices for each sample index
                for i in range(len(nbrs)):
                    self.random_state.shuffle(nbrs[i])

                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = restricted_nbr_permutation(
                    nbrs, random_seed=self.random_seed
                )

                # update the X variable column
                data_copy.loc[:, x_cols] = data.loc[restricted_permutation, x_cols].to_numpy()

                # compute the CMI on the shuffled array
                null_dist[idx] = self._compute_cmi(data_copy, x_vars, y_vars, z_covariates)
        else:
            null_dist = self._compute_shuffle_dist(
                data, x_vars, y_vars, z_covariates, n_shuffle=n_shuffle
            )

        return null_dist

    def _compute_shuffle_dist(
        self,
        data: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Set,
        n_shuffle: int,
    ) -> ArrayLike:
        """Compute a shuffled distribution of the test statistic."""
        data_copy = data.copy()
        x_cols = list(x_vars)
        # initialize the shuffle distribution
        shuffle_dist = np.zeros((n_shuffle,))
        for idx in range(n_shuffle):
            # compute a shuffled version of the data
            x_data = data[x_cols]
            shuffled_x = sklearn.utils.shuffle(x_data, random_state=self.random_seed)

            # compute now the test statistic on the shuffle data
            data_copy[x_cols] = shuffled_x.values
            shuffle_dist[idx] = self._compute_cmi(data_copy, x_vars, y_vars, z_covariates)

        return shuffle_dist

    def _compute_cmi(
        self, df: pd.DataFrame, x_vars: Set[Column], y_vars: Set[Column], z_covariates: Set[Column]
    ):
        raise NotImplementedError("All CMI methods must implement a _compute_cmi function.")
