from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.special
import sklearn
import sklearn.metrics
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from dodiscover.typing import Column

from .base import BaseConditionalIndependenceTest


class ClassifierCITest(BaseConditionalIndependenceTest):
    def __init__(
        self,
        clf: sklearn.base.BaseEstimator,
        metric: Callable = sklearn.metrics.roc_auc_score,
        bootstrap: bool = False,
        n_iter: int = 20,
        correct_bias: bool = True,
        threshold: float = 0.03,
        test_size: Union[int, float] = 0.3,
        random_state: Optional[int] = None,
    ) -> None:
        """Classifier conditional independence test (CCIT).

        Implements the classifier conditional independence test in :footcite:`Sen2017model`.
        If a Z variable is not passed in, then will run a standard independence test
        using the classifier :footcite:`Lopez2016revisiting`.

        Parameters
        ----------
        clf : instance of sklearn.base.BaseEstimator
            An instance of a classification model.
        metric : Callable of sklearn metric
            A metric function to measure the performance of the classification model.
        bootstrap : bool, optional
            Whether or not to repeat runs, by default False.
        n_iter : int, optional
            If ``bootstrap`` is True, then how many iterations to run, by default 20.
        threshold : float
            The threshold to apply to reject the null hypothesis. See Notes.
        test_size : Union[int, float], optional
            The size of the teset set, by default 0.25. If less than 1, then
            will take a fraction of ``n_samples``.

        """
        self.clf = clf
        self.metric = metric
        self.correct_bias = correct_bias
        self.bootstrap = bootstrap
        self.n_iter = n_iter
        self.threshold = threshold
        self.test_size = test_size
        self.random_state = check_random_state(random_state)

    def _unconditional_shuffle(
        self, x_arr: NDArray, y_arr: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Generate samples to emulate X independent of Y.

        Shuffles the Y samples, such that if there is any dependence among X and Y,
        they are broken and the resulting samples emulate those which came from
        the distribution with :math:`X \perp Y`.

        Parameters
        ----------
        x_arr : NDArray of shape (n_samples, n_dims_x)
            The input X variable data.
        y_arr : NDArray of shape (n_samples, n_dims_y)
            The input Y variable data.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            The X_ind, y_ind, X_joint, y_joint to be used in
            binary classification, where ``X_ind`` is the data features
            that are from the generated independent data and ``X_joint``
            is the data features from the original jointly distributed data.
            ``y_ind`` and ``y_joint`` correspond to the class labels 0 and 1
            respectively.
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
        self, x_arr: NDArray, y_arr: NDArray, z_arr: NDArray, k: int = 1
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Generate dataset for CI testing as binary classification problem.

        Implements a nearest-neighbor bootstrap approach for generating samples from
        the null hypothesis where :math:`X \perp Y | Z`.

        Parameters
        ----------
        x_arr : NDArray of shape (n_samples, n_dims_x)
            The input X variable data.
        y_arr : NDArray of shape (n_samples, n_dims_y)
            The input Y variable data.
        z_arr : NDArray of shape (n_samples, n_dims_x)
            The input Z variable data.
        k : int, optional
            _description_, by default 1

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            The X_ind, y_ind, X_joint, y_joint to be used in
            binary classification, where ``X_ind`` is the data features
            that are from the generated independent data and ``X_joint``
            is the data features from the original jointly distributed data.
            ``y_ind`` and ``y_joint`` correspond to the class labels 0 and 1
            respectively.
        """
        # compute the nearest neighbors in the space of "Z training" using ball-tree alg.
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="l2").fit(z_arr)

        # then get the K nearest nbrs in the Z space
        _, indices = nbrs.kneighbors(z_arr)

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

    def _compute_test_statistic(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        # generate training and testing data
        X_train, Y_train, X_test, Y_test = self.generate_train_test_data(
            df, x_vars, y_vars, z_covariates
        )
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

        # fit the classifier on training data
        self.clf.fit(X_train, Y_train)

        # evaluate on test data and compute metric
        Y_pred = self.clf.predict(X_test)
        metric = self.metric(Y_test, Y_pred)
        pvalue = 1.0

        if not self.correct_bias:
            if metric < 0.5 - self.threshold:
                pvalue = 0.0
            metric = metric - 0.5
        else:
            n_dims_x = len(x_vars)

            # now run CCITv2 in the paper and remove the X data
            X_train = X_train[:, n_dims_x:]
            X_test = X_test[:, n_dims_x:]

            # fit the classifier on training data
            self.clf.fit(X_train, Y_train)

            # evaluate on test data and compute metric
            Y_pred = self.clf.predict(X_test)

            biased_metric = self.metric(Y_test, Y_pred)

            if metric < biased_metric - self.threshold:
                pvalue = 0.0
            metric = metric - biased_metric
        return metric, pvalue

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        self._check_test_input(df, x_vars, y_vars, z_covariates)

        n_samples = df.shape[0]

        if self.bootstrap:
            boot_metrics = []
            boot_pvalues = []

            # run test generation on bootstrapped dataset N times
            for _ in range(self.n_iter):
                sampled_df = df.sample(
                    n=n_samples, axis=0, replace=True, random_state=self.random_state
                )
                metric, pvalue = self._compute_test_statistic(
                    sampled_df, x_vars, y_vars, z_covariates
                )

                boot_metrics.append(metric)
                boot_pvalues.append(pvalue)

            # compute final pvalue
            metric = np.mean(boot_metrics)
            std_metric = np.std(boot_metrics)
            sigma = std_metric / np.sqrt(self.n_iter)
        else:
            metric, pvalue = self._compute_test_statistic(df, x_vars, y_vars, z_covariates)
            sigma = 1 / np.sqrt(n_samples)

        # normal distribution CDF = (1 + Erf(x / \sqrt{2})) / 2
        # \Phi(x, \mu, \sigma) = 0.5 * (1 + Erf((x - \mu) / (\sigma * sqrt{2}))
        # metric here is estimated x - \mu (under the null; 0.5 or the bias corrected version)
        # and \sigma is the estimated standard deviation
        pvalue = 0.5 * scipy.special.erfc(metric / (np.sqrt(2) * sigma))

        self.stat_ = metric
        self.pvalue_ = pvalue
        return metric, pvalue

    def generate_train_test_data(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Generate a training and testing dataset for CCIT.

        This takes a conditional independence problem given a dataset
        and converts it to a binary classification problem.

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        x_vars : Set[Column]
            _description_
        y_vars : Set[Column]
            _description_
        z_covariates : Optional[Set[Column]], optional
            _description_, by default None

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
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
            # TODO: consider permuting data rows randomly first since
            # conditional_shuffle does not do that
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
                x_arr_train, y_arr_train, z_arr_train
            )
            X_train = np.vstack((X_ind, X_joint))
            Y_train = np.vstack((y_ind, y_joint))

            # generate the testing dataset
            X_ind, y_ind, X_joint, y_joint = self._conditional_shuffle(
                x_arr_test, y_arr_test, z_arr_test
            )
            X_test = np.vstack((X_ind, X_joint))
            Y_test = np.vstack((y_ind, y_joint))

        return X_train, Y_train, X_test, Y_test
