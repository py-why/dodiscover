from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dodiscover.toporder._base import BaseCAMPruning, SteinMixin
from dodiscover.toporder.utils import full_DAG


class SCORE(BaseCAMPruning, SteinMixin):
    """The SCORE algorithm :footcite:`rolland2022` for causal discovery.

    The method iteratively defines a topological ordering finding leaf nodes by comparison of the
    variance terms of the diagonal entries of the Hessian of the log likelihood matrix.
    Then it prunes the fully connected DAG with CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.

    Parameters
    ----------
    eta_G: float
        Regularization parameter for Stein gradient estimator
    eta_H : float
        Regularization parameter for Stein Hessian estimator
    cam_cutoff : float
        alpha value for independence testing for edge pruning
    n_splines : int
        Default number of splines to use for the feature function.
        Automatically decreased in case of insufficient samples
    splines_degree: int
        Order of spline to use for the feature function
    estimate_variance : bool
        If True, store estimates the variance of the noise terms from the diagonal of
        Stein Hessian estimator
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
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        cam_cutoff: float = 0.001,
        n_splines: int = 10,
        splines_degree: int = 3,
        estimate_variance=False,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        super().__init__(
            cam_cutoff, n_splines, splines_degree, pns, pns_num_neighbors, pns_threshold
        )
        self.eta_G = eta_G
        self.eta_H = eta_H
        self.var = list()  # data structure for estimated variance of SCM noise terms
        self.estimate_variance = estimate_variance

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

        def _estimate_var(stein_instance, data):
            H_diag = stein_instance.hessian_diagonal(data, self.eta_G, self.eta_H)
            H_diag = H_diag / H_diag.mean(axis=0)
            return 1 / H_diag[:, 0].var(axis=0).item()

        _, d = X.shape
        order = []
        active_nodes = list(range(d))
        stein = SteinMixin()
        for _ in range(d - 1):
            X = self._find_leaf_iteration(stein, X, active_nodes, order)
        order.append(active_nodes[0])
        if self.estimate_variance:
            self.var.append(_estimate_var(stein, X))
        order.reverse()
        return full_DAG(order), order

    def prune(self, X, A_dense: NDArray) -> NDArray:
        """SCORE pruning of the fully connected adj. matrix representation of the inferred order.

        If self.do_pns = True or self.do_pns is None and number of nodes >= 20, then
        Preliminary Neighbors Search :footcite:`Buhlmann2013` is applied before CAM pruning.
        """
        d = A_dense.shape[0]
        if (self.do_pns) or (self.do_pns is None and d > 20):
            A_dense = self.pns(A_dense, X=X)
        return super().prune(X, A_dense)

    def _find_leaf_iteration(
        self, stein_instance: SteinMixin, X: NDArray, active_nodes: List[int], order: List[int]
    ) -> NDArray:
        """Find a leaf by inspection of the diagonal elements of the Hessian of the log-likelihood.

        Leaf node equals the 'argmin' of the variance of the diagonal terms of the Hessian of
        the log-likelihood.
        After a leaf is identified, it is added to the topological order, and the list of nodes
        without position in the ordering is updated.

        Parameters
        ----------
        stein_instance : SteinMixin
            Instance of the Stein estimator with helper methods.
        X : np.ndarray
            Matrix of the data (active nodes only).
        active_nodes : List[int]
            List of the nodes without a position in the topological ordering.
        order : List[int]
            List of partial order.

        Return
        ------
        X : np.ndarray
            Matrix of the data without the column corresponding to the identified leaf node.
        """
        H_diag = stein_instance.hessian_diagonal(X, self.eta_G, self.eta_H)
        leaf = int(H_diag.var(axis=0).argmin())
        leaf = self.get_leaf(
            leaf, active_nodes, order
        )  # get leaf consistent with order imposed by included_edges from self.context
        order.append(active_nodes[leaf])
        active_nodes.pop(leaf)
        X = np.hstack([X[:, 0:leaf], X[:, leaf + 1 :]])
        if self.estimate_variance:
            self.var.append(1 / H_diag[:, leaf].var(axis=0).item())
        return X
