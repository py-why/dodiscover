from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict

from dodiscover.toporder._base import BaseTopOrder, SteinMixin
from dodiscover.toporder.utils import full_dag, pns


class NoGAM(BaseTopOrder, SteinMixin):
    """The NoGAM (Not only Gaussian Additive Model) algorithm for causal discovery.

    NoGAM :footcite:`Montagna2023b` iteratively defines a topological ordering finding leaf nodes by
    predicting the entries in the gradient of the log-likelihood via estimated residuals.
    Then it prunes the fully connected DAG with CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model, while it doesn't need to assume any distribution
    of the noise terms.

    Parameters
    ----------
    n_crossval : int, optional
        Residuals of each variable in the graph are estimated via KernelRidgeRegressor of
        the sklearn library.
        To avoid overfitting in the prediction of the residuals, the method uses leave out
        cross validation, training a number of models equals `n_crossval`, which is used
        to predict the residuals on the portion of validation data unseen during the fitting
        of the regressor. Default value is 5.
        Similarly, KernelRidgeRegressor with 'rbf' kernel is used to predict entries in the
        gradient of the log-likelihood via estimated residuals.
    ridge_alpha: float, optional
        Alpha value for KernelRidgeRegressor with 'rbf' kernel, default is 0.01.
        ridge_alpha is used to fit both the regressor for the residuals estimation
        (Equation (14) :footcite:`Montagna2023b`) and for the estimation of the score entries
        from the estimated residuals.
    ridge_gamma: float, optional
        Gamma value for KernelRidgeRegressor with 'rbf' kernel, default is 0.1.
        ridge_gamma is used to fit both the regressor for the residuals estimation
        (Equation (20) :footcite:`Montagna2023b`) and for the estimation of the score entries
        from the estimated residuals.
    eta_G: float, optional
        Regularization parameter for Stein gradient estimator, default is 0.001.
    eta_H : float, optional
        Regularization parameter for Stein Hessian estimator, default is 0.001.
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.05.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10.
        Automatically decreased in case of insufficient samples
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    pns : bool, optional
        If True, perform Preliminary Neighbour Search (PNS) before CAM pruning step,
        default is False. Allows scaling CAM pruning and ordering to large graphs.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Prior knowledge about the included and excluded directed edges in the output DAG
    is supported. It is not possible to provide explicit constraints on the relative
    positions of nodes in the topological ordering. However, explicitly including a
    directed edge in the DAG defines an implicit constraint on the relative position
    of the nodes in the topological ordering (i.e. if directed edge `(i,j)` is
    encoded in the graph, node `i` will precede node `j` in the output order).
    """

    def __init__(
        self,
        n_crossval: int = 5,
        ridge_alpha: float = 0.01,
        ridge_gamma: float = 0.1,
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        alpha: float = 0.05,
        prune: bool = True,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        super().__init__(
            alpha, prune, n_splines, splines_degree, pns, pns_num_neighbors, pns_threshold
        )
        self.eta_G = eta_G
        self.eta_H = eta_H
        self.n_crossval = n_crossval
        self.ridge_alpha = ridge_alpha
        self.ridge_gamma = ridge_gamma

    def _top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        """Find the topological ordering of the causal variables from X dataset.

        Parameters
        ----------
        X : np.ndarray
            Dataset with n x d observations of the causal variables

        Returns
        -------
        A_dense : np.ndarray
            Fully connected matrix admitted by the topological ordering
        order : List[int]
            Inferred causal order
        """
        _, d = X.shape
        top_order: List[int] = list()

        remaining_nodes = list(range(d))
        for _ in range(d - 1):
            S = self.score(X[:, remaining_nodes], eta_G=self.eta_G)
            R = self._estimate_residuals(X[:, remaining_nodes])
            err = self._mse(R, S)
            leaf = np.argmin(err)
            leaf = self._get_leaf(
                leaf, remaining_nodes, top_order
            )  # get leaf consistent with order imposed by included_edges from self.context
            l_index = remaining_nodes[leaf]
            top_order.append(l_index)
            remaining_nodes = remaining_nodes[:leaf] + remaining_nodes[leaf + 1 :]

        top_order.append(remaining_nodes[0])
        top_order = top_order[::-1]
        return full_dag(top_order), top_order

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """Pruning of the fully connected adjacency matrix representation of the
        inferred topological order.

        If self.do_pns = True or self.do_pns is None and number of nodes >= 20, then
        Preliminary Neighbors Search is applied before CAM pruning.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        A_dense : np.ndarray of shape (n_nodes, n_nodes)
            Dense adjacency matrix to be pruned.

        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        d = A_dense.shape[0]
        if (self.do_pns) or (self.do_pns is None and d > 20):
            A_dense = pns(
                A=A_dense,
                X=X,
                pns_threshold=self.pns_threshold,
                pns_num_neighbors=self.pns_num_neighbors,
            )
        return super()._prune(X, A_dense)

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

        Returns
        -------
        err : np.array of shape (n_nodes, )
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
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.

        Returns
        -------
        R : np.ndarray of shape (n_samples, n_nodes)
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
