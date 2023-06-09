from typing import Set, Tuple
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import r2_score

from dodiscover.typing import Column

from .base import BaseConditionalDiscrepancyTest


def invariant_residual_test(
    X,
    Y,
    z,
    method="gam",
    test="ks",
    method_kwargs={},
    return_model=False,
    combine_pvalues=True,
):
    r"""
    Calulates the 2-sample test statistic.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Features to condition on
    Y : ndarray, shape (n,)
        Target or outcome features
    z : list or ndarray, shape (n,)
        List of zeros and ones indicating which samples belong to
        which groups.
    method : {"forest", "gam", "linear"}, default="gam"
        Method to predict the target given the covariates
    test : {"whitney_levene", "ks"}, default="ks"
        Test of the residuals between the groups
    method_kwargs : dict
        Named arguments to pass to the prediction method.
    return_model : boolean, default=False
        If true, returns the fitted model
    combine_pvalues: bool, default=True
        If True, returns hte minimum of the corrected pvalues.

    Returns
    -------
    pvalue : float
        The computed *k*-sample p-value.
    r2 : float
        r2 score of the regression fit
    model : object
        Fitted regresion model, if return_model is True
    """

    if method == "forest":
        from sklearn.ensemble import RandomForestRegressor

        predictor = RandomForestRegressor(max_features="sqrt", **method_kwargs)
    elif method == "gam":
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import SplineTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV

        pipe = Pipeline(
            steps=[
                ("spline", SplineTransformer(n_knots=4, degree=3)),
                ("linear", LinearRegression(**method_kwargs)),
            ]
        )
        param_grid = {
            "spline__n_knots": [3, 5, 7, 9],
        }
        predictor = GridSearchCV(
            pipe, param_grid, n_jobs=-2, refit=True,
            scoring="neg_mean_squared_error"
        )
    elif method == "linear":
        from sklearn.linear_model import LinearRegression

        predictor = LinearRegression(**method_kwargs)
    else:
        raise ValueError(f"Method {method} not a valid option.")

    predictor = predictor.fit(X, Y)
    Y_pred = predictor.predict(X)
    residuals = Y - Y_pred
    r2 = r2_score(Y, Y_pred)

    if test == "whitney_levene":
        from scipy.stats import mannwhitneyu
        from scipy.stats import levene

        _, mean_pval = mannwhitneyu(
            residuals[np.asarray(z, dtype=bool)],
            residuals[np.asarray(1 - z, dtype=bool)],
        )
        _, var_pval = levene(
            residuals[np.asarray(z, dtype=bool)],
            residuals[np.asarray(1 - z, dtype=bool)],
        )
        # Correct for multiple tests
        if combine_pvalues:
            pval = min(mean_pval * 2, var_pval * 2, 1)
        else:
            pval = (min(mean_pval * 2, 1), min(var_pval * 2, 1))
    elif test == "ks":
        from scipy.stats import kstest

        _, pval = kstest(
            residuals[np.asarray(z, dtype=bool)],
            residuals[np.asarray(1 - z, dtype=bool)],
        )
    else:
        raise ValueError(f"Test {test} not a valid option.")

    if return_model:
        return pval, r2, predictor
    else:
        return pval, r2
    

class ResidualCDTest(BaseConditionalDiscrepancyTest):

    def __init__(self, method='gam', test_method='ks'):
        super().__init__()
        self.method = method
        self.test_method = test_method

    def _statistic(self, Y, group_ind, X = None) -> float:
        return super()._statistic(Y, group_ind, X)

    def test(self, df, group_col: Set[Column], y_vars: Set[Column], x_vars: Set[Column]) -> Tuple[float, float]:
        X = df[list(x_vars)].values
        Y = df[list(y_vars)].values
        z = df[list(group_col)].values

        if x_vars == set():
            from scipy.stats import kstest

            stat, pval = kstest(Y[z==1], Y[z==0])
        else:
            pval, r2 = invariant_residual_test(
                X,
                Y,
                z,
                method=self.method,
                test=self.test_method,
                method_kwargs={},
                return_model=False,
                combine_pvalues=True,
            )
            stat = r2
        return stat, pval