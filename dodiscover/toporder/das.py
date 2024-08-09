from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ttest_ind

from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import full_adj_to_order


class DAS(SCORE):
    """The DAS (Discovery At Scale) algorithm for causal discovery.

    DAS :footcite:`Montagna2023a` infer the topological ordering using
    SCORE :footcite:`rolland2022`.
    Then it finds edges in the graph by inspection of the non diagonal entries of the
    Hessian of the log likelihood.
    A final, computationally cheap, pruning step is performed with
    CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.
    DAS is a highly scalable method, allowing to run causal discovery on thousands of nodes.
    It reduces the computational complexity of the pruning method of an order of magnitude with
    respect to SCORE.

    Parameters
    ----------
    eta_G: float, optional
        Regularization parameter for Stein gradient estimator, default is 0.001.
    eta_H : float, optional
        Regularization parameter for Stein Hessian estimator, default is 0.001.
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.05.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
        If False, DAS is equivalent to SCORE.
    das_cutoff : float, optional
        Alpha value for hypothesis testing in preliminary DAS pruning.
        If None (default), it is set equal to `alpha`.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10.
        Automatically decreased in case of insufficient samples
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    min_parents : int, optional
        Minimum number of edges retained by DAS preliminary pruning step, default is 5.
        min_parents <= 5 doesn't significantly affects execution time, while increasing the
        accuracy.
    max_parents : int, optional
        Maximum number of parents allowed for a single node, default is 20.
        Given that CAM pruning is inefficient for > ~20 nodes, larger values are not advised.
        The value of max_parents should be decrease under the assumption of sparse graphs.

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
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        alpha: float = 0.05,
        prune: bool = True,
        das_cutoff: Optional[float] = None,
        n_splines: int = 10,
        splines_degree: int = 3,
        min_parents: int = 5,
        max_parents: int = 20,
    ):
        super().__init__(
            eta_G, eta_H, alpha, prune, n_splines, splines_degree, estimate_variance=True, pns=False
        )
        self.min_parents = min_parents
        self.max_parents = max_parents
        self.das_cutoff = alpha if das_cutoff is None else das_cutoff

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """
        DAS preliminary pruning of A_dense matrix representation of a fully connected graph.

        Parameters
        ----------
        X : np.ndarray
            n x d matrix of the data
        A_dense : np.ndarray
            fully connected matrix corresponding to a topological ordering

        Returns
        -------
        np.ndarray
            Sparse adjacency matrix representing the pruned DAG.
        """
        _, d = X.shape
        order = full_adj_to_order(A_dense)
        max_parents = self.max_parents + 1  # +1 to account for A[l, l] = 1
        remaining_nodes = list(range(d))
        A_das = np.zeros((d, d))

        hess = self.hessian(X, eta_G=self.eta_G, eta_H=self.eta_H)
        for i in range(d - 1):
            leaf = order[::-1][i]
            hess_l = hess[:, leaf, :][:, remaining_nodes]
            hess_m = np.abs(np.median(hess_l * self.var[leaf], axis=0))
            max_parents = min(max_parents, len(remaining_nodes))

            # Find index of the reference for the hypothesis testing
            topk_indices = np.argsort(hess_m)[::-1][:max_parents]
            topk_values = hess_m[topk_indices]  # largest
            argmin = topk_indices[np.argmin(topk_values)]

            # Edges selection step
            parents = []
            hess_l = np.abs(hess_l)
            l_index = remaining_nodes.index(
                leaf
            )  # leaf index in the remaining nodes (from 0 to len(remaining_nodes)-1)
            for j in range(max_parents):
                node = topk_indices[j]
                if node != l_index:  # enforce diagonal elements = 0
                    if j < self.min_parents:  # do not filter minimum number of parents
                        parents.append(remaining_nodes[node])
                    else:  # filter potential parents with hp testing
                        # Use hess_l[:, argmin] as sample from a zero mean population
                        # (implicit assumption: argmin corresponds to zero mean hessian entry)
                        _, pval = ttest_ind(
                            hess_l[:, node],
                            hess_l[:, argmin],
                            alternative="greater",
                            equal_var=False,
                        )
                        if pval < self.das_cutoff:
                            parents.append(remaining_nodes[node])

            A_das[parents, leaf] = 1
            remaining_nodes.pop(l_index)

        return super()._prune(X, A_das)
