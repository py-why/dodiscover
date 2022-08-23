from math import log, sqrt
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

from dodiscover.typing import Column

from .base import BaseConditionalIndependenceTest


class ClassifierCITest(BaseConditionalIndependenceTest):
    def __init__(
        self, clf: sklearn.base.BaseEstimator, bootstrap: bool = False, n_iter: int = 30
    ) -> None:
        self.clf = clf

    def _conditional_knn(
        self,
    ):
        pass

    def test(
        self,
        df: pd.DataFrame,
        x_var: Column,
        y_var: Column,
        z_covariates: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        pass
