from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression

from dodiscover.ci.kernel_utils import _default_regularization
from dodiscover.typing import Column


class BaseConditionalDiscrepancyTest(metaclass=ABCMeta):
    """Abstract class for any conditional discrepancy test.

    All CD tests are used in constraint-based causal discovery algorithms. This
    class interface is expected to be very lightweight to enable anyone to convert
    a function for CD testing into a class, which has a specific API.
    """

    n_jobs: Optional[int]
    propensity_est: Optional[ArrayLike]
    propensity_model: Optional[Callable]

    def _check_test_input(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        group_col: Column,
    ):
        if any(col not in df.columns for col in x_vars):
            raise ValueError("The x variables are not all in the DataFrame.")
        if any(col not in df.columns for col in y_vars):
            raise ValueError("The y variables are not all in the DataFrame.")
        if group_col not in df.columns:
            raise ValueError(f"The group column {group_col} is not in the DataFrame.")

        if self.propensity_model is not None and self.propensity_est is not None:
            raise ValueError(
                "Both propensity model and propensity estimates are specified. Only one is allowed."
            )
        if self.propensity_est is not None:
            if self.propensity_est.shape[0] != len(df[group_col]):
                raise ValueError(
                    f"There are {self.propensity_est.shape[0]} pre-defined estimates, while "
                    f"there are {len(df[group_col])} unique groups."
                )
            if self.propensity_est.shape[1] != len(df[group_col].unique()):
                raise ValueError(
                    f"There are {self.propensity_est.shape[1]} group pre-defined estimates, while "
                    f"there are {len(df[group_col].unique())} samples."
                )

    @abstractmethod
    def test(
        self, df: pd.DataFrame, x_vars: Set[Column], y_vars: Set[Column], group_col: Column
    ) -> Tuple[float, float]:
        """Abstract method for all conditional discrepancy tests.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        x_vars : Set of column
            A column in ``df``.
        y_vars : Set of column
            A column in ``df``.
        group_col : column
            A column in ``df`` that indicates which group of distribution
            each sample belongs to with a '0', or '1'.

        Returns
        -------
        Tuple[float, float]
            Test statistic and pvalue.
        """
        pass

    def _compute_propensity_scores(self, group_ind, **kwargs):
        if self.propensity_model is None:
            K = kwargs.get("K")

            # compute a default penalty term if using a kernel matrix
            if K.shape[0] == K.shape[1]:
                self.propensity_penalty_ = _default_regularization(K)
                C = 1 / (2 * self.propensity_penalty_)
            else:
                self.propensity_penalty_ = 0.0
                C = 1.0

            # default model is logistic regression
            self.propensity_model_ = LogisticRegression(
                penalty="l2",
                n_jobs=self.n_jobs,
                warm_start=True,
                solver="lbfgs",
                random_state=self.random_state,
                C=C,
            )
        else:
            self.propensity_model_ = self.propensity_model

        # either use pre-defined propensity weights, or estimate them
        if self.propensity_est is None:
            K = kwargs.get("K")
            # fit and then obtain the probabilities of treatment
            # for each sample (i.e. the propensity scores)
            self.propensity_est_ = self.propensity_model_.fit(K, group_ind).predict_proba(K)[:, 1]
        else:
            self.propensity_est_ = self.propensity_est[:, 1]
        return self.propensity_est_

    @abstractmethod
    def _statistic(self, X: ArrayLike, Y: ArrayLike, group_ind: ArrayLike) -> float:
        """Abstract method for computing the test statistic."""
        pass

    def compute_null(
        self, e_hat: ArrayLike, X: ArrayLike, Y: ArrayLike, null_reps: int = 1000, random_state=None
    ) -> ArrayLike:
        """Estimate null distribution using propensity weights.

        Parameters
        ----------
        e_hat : Array-like of shape (n_samples,)
            The predicted propensity score for ``group_ind == 1``.
        X : Array-Like of shape (n_samples, n_features_x)
            The X (covariates) array.
        Y : Array-Like of shape (n_samples, n_features_y)
            The Y (outcomes) array.
        null_reps : int, optional
            Number of times to sample null, by default 1000.
        random_state : int, optional
            Random generator, or random seed, by default None.

        Returns
        -------
        null_dist : Array-like of shape (n_samples,)
            The null distribution of test statistics.
        """
        rng = np.random.default_rng(random_state)

        n_samps = X.shape[0]

        # compute the test statistic on the conditionally permuted
        # dataset, where each group label is resampled for each sample
        # according to its propensity score
        null_dist = Parallel(n_jobs=self.n_jobs)(
            [
                delayed(self._statistic)(X, Y, rng.binomial(1, e_hat, size=n_samps))
                for _ in range(null_reps)
            ]
        )
        return null_dist
