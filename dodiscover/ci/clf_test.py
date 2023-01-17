from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.special
import sklearn
import sklearn.metrics
from sklearn.utils import check_random_state

from dodiscover.typing import Column

from .base import BaseConditionalIndependenceTest, ClassifierCIMixin
from .typing import Classifier


class ClassifierCITest(BaseConditionalIndependenceTest, ClassifierCIMixin):
    def __init__(
        self,
        clf: Classifier,
        metric: Callable = sklearn.metrics.accuracy_score,
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
        clf : instance of sklearn.base.BaseEstimator or pytorch model
            An instance of a classification model. If a PyTorch model is used,
            then the user must pass the PyTorch model through ``skorch`` to turn
            the Neural Network into an object that is sklearn-compliant API.
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
        random_state : int, optional
            The random seed that is used to seed via ``np.random.defaultrng``.

        Notes
        -----
        A general problem with machine-learning prediction based approaches is, that they
        don't find all kind of dependencies, only when they impact the expectation. For instance,
        a dependency with respect to the variance would not be captured by the CCIT:

        .. code-block:: python
            import numpy as np
            from dowhy.gcm import kernel_based, regression_based

            X = np.random.normal(0, 1, 1000)
            Y = []

            for x in X:
                Y.append(np.random.normal(0, abs(x)))

            Y = np.array(Y)
            Z = np.random.normal(0, 1, 1000)

            print("Correct result:", kernel_based(X, Y, Z))
            print("Wrong result", regression_based(X, Y, Z))

            clf = RandomForestClassifier()
            ci_estimator = ClassifierCITest(clf)

            df = pd.DataFrame({'x': X, 'y': Y, 'z': Z})

            _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
            print("Wrong result", pvalue)

        References
        ----------
        .. footbibliography::
        """
        self.clf = clf
        self.metric = metric
        self.correct_bias = correct_bias
        self.bootstrap = bootstrap
        self.n_iter = n_iter
        self.threshold = threshold
        self.test_size = test_size

        # set the internal random state generator
        self.random_state = check_random_state(random_state)

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
        binary_pvalue = 1.0

        if not self.correct_bias:
            if metric < 0.5 - self.threshold:
                binary_pvalue = 0.0
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
                binary_pvalue = 0.0
            metric = metric - biased_metric
        return metric, binary_pvalue

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
            std_metric = np.std(boot_metrics) + 1e-8
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
        self.stat_sigma_ = sigma
        self.pvalue_ = pvalue
        return metric, pvalue
