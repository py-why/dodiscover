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


class ClassifierCMITest(BaseConditionalIndependenceTest):
    def __init__(self, 
        clf: sklearn.base.BaseEstimator,
        metric: Callable = sklearn.metrics.accuracy_score,
        bootstrap: bool = False,
        n_iter: int = 20,
        correct_bias: bool = True,
        threshold: float = 0.03,
        test_size: Union[int, float] = 0.3,
        random_state: Optional[int] = None,) -> None:

        self.clf = clf
        self.metric = metric
        self.correct_bias = correct_bias
        self.bootstrap = bootstrap
        self.n_iter = n_iter
        self.threshold = threshold
        self.test_size = test_size
        self.random_state = check_random_state(random_state)


    def test(self, df: pd.DataFrame, x_vars: Set[Column], y_vars: Set[Column], z_covariates: Optional[Set[Column]] = None) -> Tuple[float, float]:
        x_var = x_vars[0]
        y_var = y_vars[0]

        # actually compute the cmi estimated value
        cmi_est = self._compute_cmi(df, x_var, y_var, z_covariates)
        if max(0, cmi_est) < self.threshold:
            pvalue = 0.0
        else:
            pvalue = 1.0  

        return cmi_est, pvalue

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