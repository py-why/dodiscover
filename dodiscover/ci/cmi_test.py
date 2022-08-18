from typing import Any, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import scipy.special
from scipy.stats import norm

from .base import BaseConditionalIndependenceTest


class CMITest(BaseConditionalIndependenceTest):
    def __init__(self, k=0.2, n_shuffle_nghbrs=5, transform='rank', n_jobs=-1, ) -> None:
        self.k=k

    def test(self, df: pd.DataFrame, x_var: Any, y_var: Any, z_covariates: Any = None) -> Tuple[float, float]:
        n_samples, n_dims = df.shape

        if self.k < 1:
            knn_here = max(1, int(self.k*n_samples))
        else:
            knn_here = max(1, int(self.k))
        
        # compute the K nearest neighbors in sub-spaces
        k_xz, k_yz, k_z = self._get_knn(df, x_var, y_var, z_covariates)

        # compute the final CMI value
        val = scipy.special.digamma(knn_here) - (scipy.special.digamma(k_xz) +
                                           scipy.special.digamma(k_yz) -
                                           scipy.special.digamma(k_z)).mean()

        # compute the significance of the CMI value

        self.stat_ = val

    def _get_knn(self, data, x_var, y_var, z_covariates):
        pass

    def _compute_shuffle_sig(self, data, x_var, y_var, z_covariates, value):
        pass