import math
import sys

import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler

from .base import BaseConditionalIndependenceTest


class PartialCorrelation(BaseConditionalIndependenceTest):
    def __init__(
        self,
        method="analytic",
        fixed_threshold=0.1,
        bootstrap_n_samples=1000,
        random_state=None,
        block_length=1,
        verbose=False,
    ):
        self.method = method
        self.fixed_threshold = fixed_threshold
        self.bootstrap_n_samples = bootstrap_n_samples
        self.block_length = block_length
        self.verbose = verbose
        if random_state is None:
            random_state = np.random.RandomState()
        else:
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

    def test(self, df, x_var, y_var, z_covariates=None):
        """Perform CI test of X, Y given optionally Z.

        Parameters
        ----------
        x : str | int
            The first variable in the column of ``df``.
        y : str | int
            The second variable in the column of ``df``.
        z : str | int
            The conditioning dataset, by default None.
        """
        self._check_test_input(df, x_var, y_var, z_covariates)

        if z_covariates is None or len(z_covariates) == 0:
            Z = None
        else:
            z_covariates = list(z_covariates)
            Z = df[z_covariates].to_numpy().reshape((-1, len(z_covariates)))
        X = df[x_var].to_numpy()[:, np.newaxis]
        Y = df[y_var].to_numpy()[:, np.newaxis]

        # stack the X and Y arrays
        XY = np.hstack((X, Y))

        # handle if conditioning set is passed in or not
        if Z is None:
            data_arr = XY
        else:
            data_arr = np.hstack((XY, Z))

        # Ensure it is a valid array
        if np.isnan(data_arr).sum() != 0:
            raise ValueError("nans in the data array.")

        if self.method == "pingouin":
            import pandas as pd
            import pingouin as pg

            if Z is not None:
                z_dims = Z.shape[1]
                covar = [f"z{idx}" for idx in range(z_dims)]
                z_list = covar
            else:
                covar = None
                z_list = []

            df = pd.DataFrame(data_arr)
            columns = ["x", "y"]
            columns.extend(z_list)
            df.columns = columns
            stats = pg.partial_corr(df, x="x", y="y", covar=covar)
            return stats["r"].values[0], stats["p-val"].values[0]

        # compute the dependence measure of the data vs indicator function
        val = self._compute_parcorr(data_arr, xvalue=0, yvalue=1)

        # get the dimension of the dataset
        n_samples, n_dims = data_arr.shape

        # compute the pvalue
        pvalue = self.compute_significance(val, data_arr, n_samples, n_dims)

        return val, pvalue

    def compute_significance(self, val, array, n_samples, n_dims, sig_override=None):
        """Compute pvalue of the partial correlation using bootstrap sampling.

        Returns the p-value from whichever significance function is specified
        for this test.  If an override is used, then it will call a different
        function then specified by self.significance

        Parameters
        ----------
        val : float
            Test statistic value.
        array : array-like
            data array with X, Y, Z in rows and observations in columns
        n_samples : int
            Sample length
        n_dims : int
            Dimensionality, ie, number of features.
        sig_override : string
            Must be in 'analytic', 'shuffle_test', 'fixed_thres'

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Defaults to the self.significance member value
        use_sig = self.method
        if sig_override is not None:
            use_sig = sig_override

        # Check if we are using the analytic significance
        if use_sig == "analytic":
            pval = self._compute_analytic_significance(
                value=val, n_samples=n_samples, n_dims=n_dims
            )
        # Check if we are using the shuffle significance
        elif use_sig == "shuffle_test":
            pval = self._compute_shuffle_significance(array=array, value=val)
        # Check if we are using the fixed_thres significance
        elif use_sig == "fixed_threshold":
            pval = self._compute_fixed_threshold_significance(
                value=val, fixed_threshold=self.fixed_threshold
            )
        else:
            raise ValueError("%s not known." % self.method)

        # Return the calculated value
        return pval

    def _compute_parcorr(self, array, xvalue, yvalue):
        """Compute the partial correlation."""
        # compute residuals when regressing Z on X and Z on Y
        x_resid = self._compute_ols_residuals(array, target_var=xvalue)
        y_resid = self._compute_ols_residuals(array, target_var=yvalue)

        # then compute the correlation using Pearson method
        val, _ = scipy.stats.pearsonr(x_resid[:5], y_resid[:5])

        return val

    def _compute_ols_residuals(self, array, target_var, standardize=True, return_means=False):
        """Compute residuals of linear multiple regression.

        Performs a OLS regression of the variable indexed by ``target_var`` on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.

        Parameters
        ----------
        array : np.ndarray of shape (n_samples, n_vars)
            Data array with X, Y, Z in rows and observations in columns.

        target_var : int
            The variable to regress out conditions from. This should
            be the value of the X or Y indicator (0, or 1) in this case
            indicating the row in ``array`` for those two datas.

        standardize : bool, optional
            Whether to standardize the array beforehand. Must be used for
            partial correlation. Default is True.

        return_means : bool, optional
            Whether to return the estimated regression line. Default is False.

        Returns
        -------
        resid : np.ndarray of shape (n_samples)
            The residual of the regression and optionally the estimated line.
        mean : np.ndarray of shape
        """
        _, dim = array.shape
        dim_z = dim - 2

        # standardize with z-score transformation
        if standardize:
            scaler = StandardScaler()
            array = scaler.fit_transform(array)

        y = array[:, target_var]
        if dim_z > 0:
            # get the (n_samples, zdim) array
            z = array[:, 2:].copy()

            # compute the least squares regression of z @ \beta = y
            # - y is a (n_samples, ydim) array
            # - beta is a (zdim, ydim) array of values
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]

            # compute the residuals of the model predictions vs actual values
            y_pred = np.dot(z, beta_hat)
            resid = y - y_pred
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        return resid

    def _compute_shuffle_significance(self, array, value, return_null_dist=False):
        """Returns p-value for shuffle significance test.

        For residual-based test statistics only the residuals are shuffled.

        Parameters
        ----------
        array : np.ndarray of shape (n_samples, n_dims)
            Data array with X, Y, Z in columns and observations in rows.
        xyz : array of ints
            XYZ identifier array of shape (dim,).
        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """

        x_vals = self._compute_ols_residuals(array, target_var=0)
        y_vals = self._compute_ols_residuals(array, target_var=1)
        array_resid = np.array([x_vals, y_vals])
        xyz_resid = np.array([0, 1])

        null_dist = self._get_shuffle_dist(
            array_resid,
            xyz_resid,
            sig_samples=self.bootstrap_n_samples,
            sig_blocklength=self.block_length,
        )

        pval = (null_dist >= np.abs(value)).mean()

        # Adjust p-value for two-sided measures
        # if pval < 1.0:
        #     pval *= 2.0

        if return_null_dist:
            return pval, null_dist
        return pval

    def _get_shuffle_dist(
        self,
        array,
        xyz,
        sig_samples,
        sig_blocklength=None,
    ):
        """Returns shuffle distribution of test statistic.

         The rows in array corresponding to the X-variable are shuffled using
         a block-shuffle approach.

         Parameters
         ----------
         array : np.ndarray of shape (n_samples, n_dims)
            Data array with X, Y, Z in columns and observations in rows.

         xyz : np.ndarray of shape (n_xyz_dims)
            XYZ identifier array of shape (dim,).

        dependence_measure : object
            Dependence measure function must be of form
            dependence_measure(array, xyz) and return a numeric value

         sig_samples : int, optional (default: 100)
            Number of samples for shuffle significance test.

         sig_blocklength : int, optional (default: None)
            Block length for block-shuffle significance test. If None, the
            block length is determined from the decay of the autocovariance as
            explained in [1]_.

         Returns
         -------
         null_dist : array of shape (sig_samples,)
             Contains the sorted test statistic values estimated from the
             shuffled arrays.
        """
        n_samples, n_dims = array.shape

        # x_indices = np.where(xyz == 0)[0]
        # dim_x = len(x_indices)
        # if sig_blocklength is None:
        #     sig_blocklength = self._get_block_length(array, xyz, mode="significance")

        n_blks = int(math.floor(float(n_samples) / sig_blocklength))
        if self.verbose:
            print("Significance test with block-length = %d " "..." % (sig_blocklength))

        array_shuffled = np.copy(array)

        # new number of rows
        M = array_shuffled.shape[0] // n_blks

        # initialize the null distribution
        null_dist = np.zeros(sig_samples)

        for idx in range(sig_samples):
            # now create a shuffling of the array for the given number of blocks
            array_shuffled = array_shuffled.reshape(M, -1, n_dims)[
                np.random.permutation(M)
            ].reshape(-1, n_dims)

            # do not shuffle non x-indices columns
            array_shuffled[:, 1:] = array[:, 1:].copy()

            # compute the test statistic null distribution
            null_dist[idx] = self._compute_parcorr(array=array_shuffled, xvalue=0, yvalue=1)

        return null_dist

    def _compute_analytic_significance(self, value, n_samples, n_dims):
        """Analytic p-value from Student's t-test for Pearson correlation coefficient.

        Assumes two-sided correlation. If the degrees of freedom are less than
        1, numpy.nan is returned.

        Parameters
        ----------
        value : float
            Test statistic value.
        n_samples : int
            Sample length
        n_dims : int
            Dimensionality, ie, number of features.

        Returns
        -------
        pval : float | numpy.nan
            P-value.
        """
        # Get the number of degrees of freedom
        deg_f = n_samples - n_dims

        if deg_f < 1:
            pval = np.nan
        elif abs(abs(value) - 1.0) <= sys.float_info.min:
            pval = 0.0
        else:
            trafo_val = value * np.sqrt(deg_f / (1.0 - value * value))
            # Two sided significance level
            pval = scipy.stats.t.sf(np.abs(trafo_val), deg_f) * 2

        return pval

    def _compute_fixed_threshold_significance(self, value, fixed_threshold):
        """Returns signficance for thresholding test.

        Returns 0 if numpy.abs(value) is smaller than ``fixed_threshold`` and 1 else.

        Parameters
        ----------
        value : float
            Value of test statistic for unshuffled estimate.

        fixed_threshold : float
            Fixed threshold, is made positive.

        Returns
        -------
        pval : float
            Returns 0 if numpy.abs(value) is smaller than fixed_thres and 1
            else.
        """
        if np.abs(value) < np.abs(fixed_threshold):
            pval = 1.0
        else:
            pval = 0.0

        return pval
