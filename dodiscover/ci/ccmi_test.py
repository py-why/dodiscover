from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics

from dodiscover.typing import Column

from .base import BaseConditionalIndependenceTest, ClassifierCIMixin, CMIMixin
from .kernel_utils import f_divergence_score, kl_divergence_score


class ClassifierCMITest(BaseConditionalIndependenceTest, ClassifierCIMixin, CMIMixin):
    def __init__(
        self,
        clf: sklearn.base.BaseEstimator,
        metric: Callable = f_divergence_score,
        bootstrap: bool = False,
        n_iter: int = 20,
        threshold: float = 0.03,
        test_size: Union[int, float] = 0.3,
        n_jobs: int = -1,
        n_shuffle_nbrs: int = 5,
        n_shuffle: int = 100,
        eps: float = 1e-8,
        random_seed: Optional[int] = None,
    ) -> None:
        """Classifier based CMI estimation for CI testing.

        Parameters
        ----------
        clf : sklearn.base.BaseEstimator
            A scikit-learn classifier.
        metric : Callable, optional
            The metric, by default f_divergence_score
        bootstrap : bool, optional
            Whether to take bootstrap samples, by default False.
        n_iter : int, optional
            Number of bootstrap iterations, by default 20.
        test_size : Union[int, float], optional
            The size, or proportion of the samples to use for the test
            dataset, by default 0.3.
        threshold : float, optional
            Threshold to state conditional independence by binarizing the
            returned pvalue, by default 0.03. If threshold is set to ``None``,
            then the null distribution will be estimated via permutation tests
            ``n_shuffle`` times to compute a pvalue.
        n_jobs : int, optional
            The number of CPUs to use, by default -1, which corresponds to
            using all CPUs available.
        n_shuffle_nbrs : int, optional
            Number of nearest-neighbors within the Z covariates for shuffling, by default 5.
        n_shuffle : int
            The number of times to shuffle the dataset to generate the null distribution.
            By default, 1000.
        eps : float
            The epsilon term to add to all predicted values of the classifier to
            prevent divisions by 0. By default 1e-8.
        random_state : Optional[int], optional
            Random seed, by default None.

        Notes
        -----
        This implementation differs from the original reference in :footcite:`Mukherjee2020ccmi`
        because we do not have a generator and discriminator implementation.

        The test proceeds similarly to the
        :class:`dodiscocver.ci.ClassifierCITest`. The test proceeds by
        first splitting the dataset randomly into two parts. One dataset
        performs a restricted permutation. This dataset is used to label
        samples that are guaranteed to not have conditional dependence, since
        the samples are shuffled. We then train our classifier on this dataset
        and evaluate it on the other dataset. We then estimate the KL-divergence
        by comparing the evaluations on the shuffled vs unshuffled dataset.

        We estimate a metric using the classifier, such as:
        - KL-divergence
        - f-divergence

        We are in general interested in the KL-divergence because mutual information
        is a special case of KL-divergence. The ``f-divergence`` is a lower bound on
        the KL-divergence, which can be more stable in certain cases. The user may choose
        which metric to compute from the classifier output.

        **Estimating the pvalue**
        If threshold is set, then the estimated CMI value is compared with
        ``0 + threshold`` to determine if the query is "conditionally dependent" and the
        resulting pvalue returned will be either 0 or 1. If the threshold is set to ``None``,
        then the pvalue is estimated via a permutation testing step, which estimates the
        null distribution of CMI values. This may be computationally expensive due to refitting
        a new classifier for each null distribution instance ``n_shuffle`` times.
        """
        self.clf = clf
        self.metric = metric
        self.bootstrap = bootstrap
        self.n_iter = n_iter
        self.test_size = test_size

        self.threshold = threshold
        self.n_jobs = n_jobs
        self.n_shuffle = n_shuffle
        self.n_shuffle_nbrs = n_shuffle_nbrs
        self.eps = eps
        self.random_seed = random_seed
        self.random_state = np.random.default_rng(random_seed)

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        """Test conditional independence by estimating CMI.

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
        if z_covariates is None:
            z_covariates = set()

        self._check_test_input(df, x_vars, y_vars, z_covariates)
        n_samples = df.shape[0]

        if self.bootstrap:
            boot_metrics = []

            # run test generation on bootstrapped dataset N times
            for _ in range(self.n_iter):
                sampled_df = df.sample(
                    n=n_samples, axis=0, replace=True, random_state=self.random_state
                )

                # estimate CMI
                cmi_metric = self._compute_cmi(sampled_df, x_vars, y_vars, z_covariates)
                boot_metrics.append(cmi_metric)

            # compute final pvalue
            metric = np.mean(boot_metrics)
            std_metric = np.std(boot_metrics)
            sigma = std_metric / np.sqrt(self.n_iter)
        else:
            metric = self._compute_cmi(df, x_vars, y_vars, z_covariates)
            sigma = 1 / np.sqrt(n_samples)

        # actually compute the cmi estimated value
        if self.threshold is None:
            # estimate null distribution to compute a pvalue
            null_dist = self._estimate_null_dist(
                df,
                x_vars,
                y_vars,
                z_covariates,
                n_shuffle_nbrs=self.n_shuffle_nbrs,
                n_shuffle=self.n_shuffle,
            )

            # compute pvalue as the number of times our metric is smaller
            # than the resulting null distribution of metrics
            print("Computing null metric... ", metric)
            print(null_dist)
            pvalue = (null_dist >= metric).mean()
            self.null_dist_ = null_dist
        else:
            print("Computing metric... ", metric)
            if max(0, metric) < self.threshold:
                pvalue = 0.0
            else:
                pvalue = 1.0

        self.stat_ = metric
        self.stat_sigma_ = sigma
        self.pvalue_ = pvalue
        return metric, pvalue

    def _compute_cmi(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set] = None,
    ) -> float:
        """Compute test statistic, the conditional mutual information.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset
        x_vars : Set[Column]
            X variables.
        y_vars : Set[Column]
            Y variables.
        z_covariates : Optional[Set], optional
            Z conditioning variables, by default None.

        Returns
        -------
        cmi_metric : float
            The estimated CMI.
        """
        if z_covariates is not None and len(z_covariates) > 0:
            # first estimate I(X; Y, Z)
            I_xyz = self._estimate_mi(df, x_vars, y_vars.union(z_covariates))

            # next estimate I(X; Y)
            I_xy = self._estimate_mi(df, x_vars, y_vars)

            # compute CMI estimate using the difference
            cmi_metric = self._compute_cmi_with_diff(I_xyz, I_xy)
        else:
            # next estimate just the mutual information I(X; Y)
            cmi_metric = self._estimate_mi(df, x_vars, y_vars)

        return cmi_metric

    def _estimate_mi(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
    ) -> float:
        """Estimate mutual information.

        Estimates mutual information via permutation and shuffling
        classification.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset.
        x_vars : Set[Column]
            X variables.
        y_vars : Set[Column]
            Y variables.

        Returns
        -------
        metric : float
            Estimate of the mutual information, either using KL-divergence,
            or f-divergence.
        """
        # generate training and testing data by shuffling X and Y
        X_train, Y_train, X_test, Y_test = self.generate_train_test_data(
            df, x_vars, y_vars, k=self.n_shuffle_nbrs
        )
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

        # fit the classifier on training data
        self.clf.fit(X_train, Y_train)

        # evaluate on test data and compute metric
        Y_pred = self.clf.predict(X_test)

        # get the indices of Y_test being 0, or 1
        p_idx = np.argwhere(Y_test == 1)  # joint
        q_idx = np.argwhere(Y_test == 0)  # (conditionally) independent

        # get the predictions for the two distributions
        Y_pred_p = Y_pred[p_idx]
        Y_pred_q = Y_pred[q_idx]

        # estimate the metric
        kwargs = dict()
        if self.metric == kl_divergence_score:
            kwargs["eps"] = self.eps
        metric = self.metric(Y_pred_q, Y_pred_p, **kwargs)
        return metric

    def _compute_cmi_with_diff(self, I_xyz: float, I_xz: float) -> float:
        return I_xyz - I_xz
