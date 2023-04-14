from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict

from dodiscover.toporder._base import BaseCAMPruning, SteinMixin
from dodiscover.toporder.utils import full_dag


class NoGAM(BaseCAMPruning, SteinMixin):
    """The NoGAM (Not only Gaussian Additive Model) algorithm :footcite:`Montagna2023b`.

    The method iteratively defines a topological ordering finding leaf nodes by
    predicting the entries in the gradient of the log-likelihood via estimated residuals.
    Then it prunes the fully connected DAG with CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model, while it doesn't need to assume any distribution
    of the noise terms.

    Parameters
    ----------
    n_crossval : int
        Residuals of each variable in the graph are estimated via KernelRidgeRegressor of
        the sklearn library.
        To avoid overfitting in the prediction of the residuals, the method uses leave out
        cross validation, training a number of models equals `n_crossval`, which is used
        to predict the residuals on the portion of validation data unseen during the fitting
        of the regressor.
        Similarly, KernelRidgeRegressor with 'rbf' kernel is used to predict entries in the
        gradient of the log-likelihood via estimated residuals.
    ridge_alpha: float
        Alpha value for KernelRidgeRegressor with 'rbf' kernel.
    ridge_gamma: float
        Gamma value for KernelRidgeRegressor with 'rbf' kernel.
    eta_G: float
        Regularization parameter for Stein gradient estimator
    eta_H : float
        Regularization parameter for Stein Hessian estimator
    cam_cutoff : float
        alpha value for independence testing for edge pruning
    n_splines : int
        Default number of splines to use for the feature function. Automatically decreased
        in case of insufficient samples
    splines_degree: int
        Order of spline to use for the feature function
    pns : bool
        If True, perform Preliminary Neighbour Search (PNS) before CAM pruning step.
        Allows scaling CAM pruning to large graphs.
        If None, execute PNS only for graphs with strictly more than 20 nodes.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float
        Threshold to use for PNS.
    """

    def __init__(
        self,
        n_crossval: int = 5,
        ridge_alpha: float = 0.01,
        ridge_gamma: float = 0.1,
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        cam_cutoff: float = 0.001,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        super().__init__(
            cam_cutoff, n_splines, splines_degree, pns, pns_num_neighbors, pns_threshold
        )
        self.eta_G = eta_G
        self.eta_H = eta_H
        self.n_crossval = n_crossval
        self.ridge_alpha = ridge_alpha
        self.ridge_gamma = ridge_gamma

    def top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        """Find the topological ordering of the causal variables from X dataset.

        Parameter
        ---------
        X : np.ndarray
            Dataset with n x d observations of the causal variables

        Return
        ------
        A_dense : np.mdarray
            Fully connected matrix admitted by the topological ordering
        order : List[int]
            Inferred causal order
        """
        _, d = X.shape
        top_order = []

        remaining_nodes = list(range(d))
        for _ in range(d - 1):
            S = self.score(X[:, remaining_nodes], eta_G=self.eta_G)
            R = self._estimate_residuals(X[:, remaining_nodes])
            err = self._mse(R, S)
            leaf = np.argmin(err)
            leaf = self.get_leaf(
                leaf, remaining_nodes, top_order
            )  # get leaf consistent with order imposed by included_edges from self.context
            l_index = remaining_nodes[leaf]
            top_order.append(l_index)
            remaining_nodes = remaining_nodes[:leaf] + remaining_nodes[leaf + 1 :]

        top_order.append(remaining_nodes[0])
        top_order = top_order[::-1]
        return full_dag(top_order), top_order

    def prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """NoGAM pruning of the fully connected adjacency matrix representation of the
        inferred topological order.

        If self.do_pns = True or self.do_pns is None and number of nodes >= 20, then
        Preliminary Neighbors Search :footcite:`Buhlmann2013` is applied before CAM pruning.
        """
        d = A_dense.shape[0]
        if (self.do_pns) or (self.do_pns is None and d > 20):
            A_dense = self.pns(A_dense, X=X)
        return super().prune(X, A_dense)

    def _mse(self, X: NDArray, Y: NDArray) -> List[float]:
        """Predict each column of Y from X and compute the Mean Squared Error (MSE).

        Parameters
        ----------
        X : np.ndarray
            Matrix of predictors observations. Usually, the n x d matrix R of
            estimated residuals
        Y  : np.ndarray
            Matrix of the target variables Y[:, i]. Usually the n x d matrix D of the
            estimated score function

        Return
        ------
        err : np.array of shape (n_dims, )
            Vector with MSE in the prediction of score_i from residual_i
        """
        err = []
        _, d = Y.shape
        err = [
            np.mean(
                (
                    Y[:, col]
                    - cross_val_predict(
                        self._create_kernel_ridge(),
                        X[:, col].reshape(-1, 1),
                        Y[:, col],
                        cv=self.n_crossval,
                    )
                )
                ** 2
            ).item()
            for col in range(d)
        ]
        return err

    def _estimate_residuals(self, X: NDArray) -> NDArray:
        """Estimate the residuals by fitting a KernelRidge regression.

        For each variable X_j, regress X_j on all the remaining variables  of X, and
        estimate the residuals.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_dims)
            Matrix of the data.

        Return
        ------
        R : np.ndarray of shape (n_samples, n_dims)
            Matrix of the residuals estimates.
        """
        R = []
        R = [
            X[:, i]
            - cross_val_predict(
                self._create_kernel_ridge(),
                np.hstack([X[:, 0:i], X[:, i + 1 :]]),
                X[:, i],
                cv=self.n_crossval,
            )
            for i in range(X.shape[1])
        ]
        return np.vstack(R).transpose()

    def _create_kernel_ridge(self):
        return KernelRidge(kernel="rbf", gamma=self.ridge_alpha, alpha=self.ridge_alpha)
