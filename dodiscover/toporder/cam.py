from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from dodiscover.toporder._base import BaseCAMPruning
from dodiscover.toporder.utils import fullAdj2Order


class CAM(BaseCAMPruning):
    """The CAM algorithm :footcite:`Buhlmann2013` for causal discovery.

    The method iteratively defines a topological ordering by leaf additions.
    Then it prunes the fully connected DAG consistent with the inferred topological order.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.

    Parameters
    ----------
    cam_cutoff : float
        alpha value for independence testing for edge pruning
    n_splines : int
        Default number of splines to use for the feature function.
        Automatically decreased in case of insufficient samples
    splines_degree: int
        Order of spline to use for the feature function
    pns : bool
        If True, perform Preliminary Neighbour Search (PNS) before CAM pruning step.
        Allows scaling CAM pruning and ordering to large graphs.
        If None, execute PNS only for graphs with strictly more than 20 nodes.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float
        Threshold to use for PNS.
    """

    def __init__(
        self,
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
        self.inf = np.finfo(np.float32).min

    def top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        """
        Find the topological ordering of the causal variables from the dataset `X`.

        Parameter
        ---------
        X : np.ndarray
            Dataset with n x d observations of the causal variables

        Return
        ------
        A_dense : np.ndarray
            Fully connexted matrix admitted by the topological ordering
        order : List[int]
            Inferred causal order
        """

        def initialize_directed_paths(d):
            G_included = self.context.included_edges  # nx.Graph with included edges
            directed_paths = np.zeros((d, d))
            np.fill_diagonal(directed_paths, 1)
            for i in range(d):
                for j in range(d):
                    if G_included.has_edge(i, j):
                        self._update_directed_paths(i, j, directed_paths)
            return directed_paths

        _, d = X.shape
        A = nx.to_numpy_array(self.context.included_edges)
        directed_paths = initialize_directed_paths(
            d
        )  # directed_paths[i,j]=1 if there is a directed path from i to j

        score_gains, score = self._initialize_score(X, directed_paths)
        while np.sum(score_gains > self.inf) > 0:
            parent, child = np.unravel_index(
                np.argmax(score_gains, axis=None), score_gains.shape
            )  # greedily select edge with largest score contribution
            A[parent, child] = 1
            score[child] += score_gains[
                parent, child
            ]  # update score of child node given the selected edge
            self._update_acyclicity_constraints(parent, child, score_gains, directed_paths)

            # Update score column associated to child node
            self._update_score(X, A, child, score_gains, score[child])

        order = fullAdj2Order(A)
        return A, order

    def _update_score(self, X: NDArray, A: NDArray, c: int, score_gains: NDArray, score_c: float):
        """
        Update column c of score_gains matrix, where c is a node associated to a new incoming edge.

        Parameters
        ----------
        X : np.ndarray
            n x d matrix if the data
        A : np.ndarray
            Current d x d adjacency matrix
        c : int
            Column of score_gains to be updated
        score_gains : np.ndarray
            Matrix of score gains to be updated
        score_c : float
            Score of c-th node under current ordering
        """

        def valid_parent(pot_parent, current_parents):
            if pot_parent == c or pot_parent in current_parents:
                return False
            if score_gains[pot_parent, c] == self.inf:
                return False

            return True

        d = A.shape[0]
        current_parents = np.argwhere(A[:, c])
        for pot_parent in range(d):
            if valid_parent(pot_parent, current_parents):
                predictors = np.append(current_parents, [pot_parent])
                gam = self._fit_model(X[:, predictors], X[:, c].reshape(-1, 1))
                residuals = X[:, c] - gam.predict(X[:, predictors])
                gain = -np.log(np.var(residuals)) - score_c
                score_gains[pot_parent, c] = gain

    def _update_directed_paths(self, parent, child, directed_paths):
        """Update directed paths in the graph given the new (parent, child) edge.

        Parameters
        ----------
        parent : int
            Parent of the child node in the input `parent`,`child` edge
        child : int
            Child of the parent node in the input `parent`,`child` edge
        directed_paths : NDArray
            Existing directed paths in the graph.
        """
        directed_paths[
            parent, child
        ] = 1  # directed_paths[i,j]=1 if there is a directed path from i to j
        child_descendants = np.argwhere(directed_paths[child, :])
        parent_ancestors = np.argwhere(directed_paths[:, parent])
        for p in parent_ancestors:  # add path from every ancestor to every descendant
            for c in child_descendants:
                directed_paths[p, c] = 1

    def _update_acyclicity_constraints(
        self, parent: int, child: int, score_gains: NDArray, directed_paths: NDArray
    ):
        """Update the acyclicity constraints and directed paths given new (parent, child) edge.

        Add aciclicity constraints given (parent, child) edge addition,
        and update the existing directed paths.
        In order to forbid i -> j edge selection, set score_gains[i, j] = -Inf

        Parameters
        ----------
        parent : int
            Parent of the child node in the input `parent`,`child` edge
        child : int
            Child of the parent node in the input `parent`,`child` edge
        score_gains : NDArray
            Matrix of the score gain.
            score_gain[i,j] is the gain in score obtaied by addition of i -> j edge to the graph.
        directed_paths : NDArray
            Existing directed paths in the graph.
        """
        score_gains[parent, child] = self.inf  # do not select same edge twice
        score_gains[child, parent] = self.inf
        self._update_directed_paths(parent, child, directed_paths)
        score_gains[np.transpose(directed_paths == 1, (1, 0))] = self.inf

    def _initialize_score(self, X: NDArray, directed_paths: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Initialize score gains matrix and the score contribution of each node.

        If self.do_pns = True or self.do_pns is None and number of nodes >= 20.

        Parameters
        ----------
        X : np.array
            n x d matrix of the data
        directed_paths : np.ndarray
            d x d matrix encoding the directed paths in the graph.
            directed_paths[i,j]=1 if there is a directed path from i to j.

        Return
        ------
        score_gain : np.ndarray
            d x d matrix of gains. score_gain[i, j] is the additive contribute to the score
            (i.e. the gain) in adding i as parent of j
        init_score : np.ndarray
            d x 1 vector of the score contribute of each node.
            Since the initial topological ordering is empty,
            all nodes are initially treated as source.
        """
        _, d = X.shape
        G_excluded = self.context.excluded_edges  # nx.Graph with excluded edges
        if (self.do_pns) or (self.do_pns is None and d > 20):
            A_pns = self.pns(A=np.ones((d, d)), X=X)
            self.exclude_edges(A_pns)

        # Initialize matrix of score gains and vector of initial scores
        score_gains = np.zeros((d, d))
        score_gains[
            np.transpose(directed_paths == 1, (1, 0))
        ] = self.inf  # avoid cycles setting entries to -Inf according to current directed_paths
        nodes_variance = np.var(X, axis=0)
        init_score = -np.log(nodes_variance)

        # Score gain for each edge i -> j
        for i in range(d):
            for j in range(d):
                if G_excluded.has_edge(i, j):
                    score_gains[i, j] = self.inf
                elif score_gains[i, j] != self.inf:
                    gam = self._fit_model(X[:, i].reshape(-1, 1), X[:, j].reshape(-1, 1))
                    residuals = X[:, j] - gam.predict(
                        X[:, i].reshape(-1, 1)
                    )  # Overfitting? Also, recomputing gam.predict(•) is likely inefficient
                    gain = (
                        -np.log(np.var(residuals)) - init_score[j]
                    )  # np.var(residuals) different from 1/n * np.sum(np.square(residuals))...
                    score_gains[i, j] = gain
        return score_gains, init_score