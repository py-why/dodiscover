import math
import warnings
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pygam import LinearGAM, s
from pygam.terms import Term, TermList
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from dodiscover.context import Context
from dodiscover.toporder.utils import kernel_width

# -------------------- Mixin class with Stein estimators -------------------- #


class SteinMixin:
    """
    Implementation of Stein gradient estimator and Stein Hessian estimator and helper methods.
    """

    def hessian(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        """Stein estimator of the Hessian of log p(x).

        The Hessian matrix is efficiently estimated by exploitaiton of the Stein identity.
        Implements :footcite:`rolland2022`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        eta_G: float
            regularization parameter for ridge regression in Stein gradient estimator.
        eta_H: float
            regularization parameter for ridge regression in Stein hessian estimator.

        Returns
        -------
        H : np.ndarray
            Stein estimator of the Hessian matrix of log p(x).

        References
        ----------
        .. footbibliography::
        """
        _, d = X.shape
        X_diff = self._X_diff(X)
        s = kernel_width(X_diff)
        K, nablaK = self._evaluate_kernel(X_diff, evaluate_nabla=True, s=s)
        G = self.score(X, eta_G, K, nablaK)

        # Compute the Hessian by column stacked together
        H = np.stack([self._hessian_col(X_diff, G, col, eta_H, K, s) for col in range(d)], axis=1)
        return H

    def score(
        self,
        X: NDArray,
        eta_G: float,
        K: NDArray = None,
        nablaK: NDArray = None,
    ) -> NDArray:
        """Stein gradient estimator of the score, i.e. gradient log p(x).

        The Stein gradient estimator :footcite:`Li2017` exploits the Stein identity
        for efficient estimate of the score function.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        eta_G: float
            regularization parameter for ridge regression in Stein gradient estimator.
        K : np.ndarray of shape (n_samples, n_samples)
            Gaussian kernel evaluated at X, by default None. If `K` is None, it is
            computed inside of the method.
        nablaK : np.ndarray of shape (n_samples, )
            <nabla, K> evaluated dot product, by default None. If `nablaK` is None, it is
            computed inside of the method.

        Returns
        -------
        G : np.ndarray
            Stein estimator of the score function.

        References
        ----------
        .. footbibliography::
        """
        n, _ = X.shape
        if K is None:
            X_diff = self._X_diff(X)
            K, nablaK = self._evaluate_kernel(X_diff, evaluate_nabla=True)

        G = np.matmul(np.linalg.inv(K + eta_G * np.eye(n)), nablaK)
        return G

    def _hessian_col(
        self, X_diff: NDArray, G: NDArray, c: int, eta: float, K: NDArray = None, s: float = None
    ) -> NDArray:
        """Stein estimator of a column of the Hessian of log p(x)

        Parameters
        ----------
        X_diff : np.ndarray of shape (n_samples, n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        G : np.ndarray
            estimator of the score function.
        c : int
            index of the column of interest.
        eta: float
            regularization parameter for ridge regression in Stein hessian estimator
        K : np.ndarray of shape (n_samples, n_samples), optional
            Gaussian kernel evaluated at X, by default None. If `K` is None, it is
            computed inside of the method.
        s : float, optional
            Width of the Gaussian kernel, by default None. If `s` is None, it is
            computed inside of the method.

        Returns
        -------
        H_col : np.ndarray
            Stein estimator of the c-th column of the Hessian of log p(x)
        """
        n, _, _ = X_diff.shape

        # Kernel
        if s is None:
            s = kernel_width(X_diff)
        if K is None:
            K, _ = self._evaluate_kernel(X_diff, s=s)

        # Stein estimate
        Gv = np.einsum("i,ij->ij", G[:, c], G)
        nabla2vK = np.einsum("ik,ikj,ik->ij", X_diff[:, :, c], X_diff, K) / s**4
        nabla2vK[:, c] -= np.einsum("ik->i", K) / s**2
        H_col = -Gv + np.matmul(np.linalg.inv(K + eta * np.eye(n)), nabla2vK)
        return H_col

    def hessian_diagonal(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        """Stein estimator of the diagonal of the Hessian matrix of log p(x).

        Parameters
        ----------
        X : np.ndarray (n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        eta_G: float
            regularization parameter for ridge regression in Stein gradient estimator.
        eta_H: float
            regularization parameter for ridge regression in Stein hessian estimator.

        Returns
        -------
        H_diag : np.ndarray
            Stein estimator of the diagonal of the Hessian matrix of log p(x).
        """
        n, _ = X.shape
        X_diff = self._X_diff(X)

        # Score function compute
        s = kernel_width(X_diff)
        K, nablaK = self._evaluate_kernel(X_diff, evaluate_nabla=True, s=s)
        G = self.score(X, eta_G, K, nablaK)

        # Hessian compute
        nabla2K = np.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
        return -(G**2) + np.matmul(np.linalg.inv(K + eta_H * np.eye(n)), nabla2K)

    def _evaluate_kernel(
        self, X_diff: NDArray, evaluate_nabla: bool = False, s: float = None
    ) -> Tuple[NDArray, Union[NDArray, None]]:
        """
        Evaluate Gaussian kernel from data.

        Parameters
        ----------
        X_diff : np.ndarray of shape (n_samples, n_samples, n_nodes)
            Matrix of the difference between samples.
        evalaue_nabla : bool, optional
            if True evaluate <nabla, K> dot product. Default is False.
        s : float, optional
            Width of the Gaussian kernel, by default None. If `s` is None, it is
            computed inside of the method.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            evaluated gaussian kernel.
        nablaK : np.ndarray of shape (n_samples, n_nodes)
            <nabla, K> dot product.
        """
        nablaK = None
        if s is None:
            s = kernel_width(X_diff)
        K = np.exp(-np.linalg.norm(X_diff, axis=2) ** 2 / (2 * s**2)) / s
        if evaluate_nabla:
            nablaK = -np.einsum("kij,ik->kj", X_diff, K) / s**2
        return K, nablaK

    def _X_diff(self, X: NDArray):
        """For each sample of X, compute its difference with all n samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.

        Returns
        -------
        X_diff : np.ndarray of shape (n_samples, n_samples, n_nodes)
            Matrix of the difference between samples.
        """
        return np.expand_dims(X, axis=1) - X


# -------------------- Topological ordering interface -------------------- #


class TopOrderInterface(metaclass=ABCMeta):
    """
    Interface for causal discovery based on estimate of topologial ordering and DAG pruning.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame, context: Context) -> None:
        raise NotImplementedError()

    @abstractmethod
    def top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        raise NotImplementedError()

    @abstractmethod
    def prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        raise NotImplementedError()


# -------------------- Base class for CAM pruning --------------------#
class BaseCAMPruning(TopOrderInterface):
    """Class for topological order based methods with CAM pruning.

    Implementation of `TopOrderInterface` defining `fit` method for causal discovery.
    Class inheriting from `BaseCAMPruning` need to implement the `top_order` method for inference
    of the topological ordering of the nodes in the causal graph.
    The resulting fully connected matrix is pruned by `prune` method implementation of
    CAM pruning :footcite:`Buhlmann2013`.

    Parameters
    ----------
    cam_cutoff : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.001.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10. Automatically decreased
        in case of insufficient samples.
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    pns : bool, optional
        If True, perform Preliminary Neighbour Search (PNS). Default is False.
        Allows scaling CAM pruning and ordering to large graphs.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    Attributes
    ----------
    graph_ : nx.DiGraph
        Adjacency matrix representation of the inferred causal graph.
    order_ : List[int]
        Topological order of the nodes from source to leaf.

    References
    ----------
    .. footbibliography::
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
        self.cam_cutoff = cam_cutoff
        self.n_splines = n_splines
        self.degree = splines_degree
        self.do_pns = pns
        self.pns_num_neighbors = pns_num_neighbors
        self.pns_threshold = pns_threshold

        self.order_: nx.DiGraph = nx.empty_graph()
        self.graph_: List[int] = list()

    def _get_leaf(self, leaf: int, remaining_nodes: List[int], current_order: List[int]) -> int:
        """Get leaf node from the list of `remaining_nodes` without an assigned order.

        Parameters
        ----------
        leaf : int
            Leaf position in the list of `remaining_nodes`.
        remaining_nodes : List[int]
            List of nodes without a position in the order.
        current_order : List[int]
            Partial topological order.

        Returns
        -------
        leaf : int
            Leaf index in the list of graph nodes.
        """
        # descendants enforced by edges in self.context and not in the order are used as leaf
        leaf_descendants = self.order_constraints[remaining_nodes[leaf]]
        if not set(leaf_descendants).issubset(set(current_order)):
            k = 0
            while True:
                if leaf_descendants[k] not in current_order:
                    leaf = remaining_nodes.index(leaf_descendants[k])
                    break  # exit when leaf is found
                k += 1
        return leaf

    def fit(self, data_df: pd.DataFrame, context: Context) -> None:
        """
        Fit topological order based causal discovery algorithm on input data.

        Parameters
        ----------
        data_df : pd.DataFrame
            Datafame of the input data.
        context: Context
            The context of the causal discovery problem.
        """
        X = data_df.to_numpy()
        self.context = context

        # Check acyclicity condition on included_edges
        self._dag_check_included_edges()
        self.order_constraints = self._included_edges_order_constraints()

        # Inference of the causal order.
        A_dense, order = self.top_order(X)
        self.order_ = order

        # Inference of the causal graph via pruning
        A = self.prune(X, A_dense)
        self.graph_ = nx.from_numpy_array(A, create_using=nx.DiGraph)

    def prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """
        Prune the dense adjacency matrix `A_dense` from spurious edges.

        Use sparse regression over the matrix of the data `X` for variable selection over the
        edges in the dense (potentially fully connected) adjacency matrix `A_dense`

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
        _, d = X.shape
        G_excluded = self.context.excluded_edges
        G_included = self.context.included_edges
        A = np.zeros((d, d))
        for c in range(d):
            pot_parents = []
            for p in self.order_[: self.order_.index(c)]:
                if ((not G_excluded.has_edge(p, c)) and A_dense[p, c] == 1) or G_included.has_edge(
                    p, c
                ):
                    pot_parents.append(p)
            if len(pot_parents) > 0:
                parents = self._variable_selection(X[:, pot_parents], X[:, c], pot_parents, c)
                for parent in parents:
                    A[parent, c] = 1

        return A

    def _pns(self, A: NDArray, X: NDArray) -> NDArray:
        """Preliminary Neighbors Selection (PNS) pruning on adjacency matrix `A`.

        Variable selection preliminary to CAM pruning.
        PNS :footcite:`Buhlmann2013` allows to scale CAM pruning to large graphs
        (~20 or more nodes), with sensitive reduction of computational time.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix representation of a dense graph.
        X : np.ndarray of shape (n_samples, n_nodes)
            Dataset with observations of the causal variables.

        Returns
        -------
        A : np.ndarray
            Pruned adjacency matrix.

        References
        ----------
        .. footbibliography::
        """
        num_nodes = X.shape[1]
        for node in range(num_nodes):
            X_copy = np.copy(X)
            X_copy[:, node] = 0
            reg = ExtraTreesRegressor(n_estimators=500)
            reg = reg.fit(X_copy, X[:, node])
            selected_reg = SelectFromModel(
                reg,
                threshold="{}*mean".format(self.pns_threshold),
                prefit=True,
                max_features=self.pns_num_neighbors,
            )
            mask_selected = selected_reg.get_support(indices=False)

            mask_selected = mask_selected.astype(A.dtype)
            A[:, node] *= mask_selected

        return A

    def _exclude_edges(self, A: NDArray) -> NDArray:
        """Update `self.context` excluding edges not admitted in `A`.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix representation of an arbitrary graph (directe or undirected).
            If `A[i, j]`, add edge (i, j) to `self.context.excluded_edges`.
        """
        # self.context.excluded_edges: nx.Graph with excluded edges
        d = A.shape[0]
        for i in range(d):
            for j in range(d):
                if i != j and A[i, j] == 0 and (not self.context.included_edges.has_edge(i, j)):
                    self.context.excluded_edges.add_edge(i, j)

    def _dag_check_included_edges(self) -> None:
        """Check that the edges included in `self.context` does not violate DAG condition."""
        is_dag = nx.is_directed_acyclic_graph(self.context.included_edges)
        if nx.is_empty(self.context.included_edges):
            is_dag = True
        if not is_dag:
            raise ValueError("Edges included in the graph violate the acyclicity condition!")

    def _included_edges_order_constraints(self) -> Dict[int, List[int]]:
        """For each node find the predecessors enforced by the edges included in `self.context`.

        Returns
        -------
        descendants : Dict[int, List[int]]
            Dictionary with index of a node of the graph as key, list of the descendants of the
            node enforced by self.context.included_edges as value.
        """
        adj = nx.to_numpy_array(self.context.included_edges)
        d, _ = adj.shape
        descendants: Dict = {i: list() for i in range(d)}
        for row in range(d):
            for col in range(d):
                if adj[row, col] == 1:
                    row_descendants = descendants[row]
                    row_descendants.append(col)
                    descendants[row] = row_descendants
        return descendants

    def _variable_selection(
        self, X: NDArray, y: NDArray, pot_parents: List[int], child: int
    ) -> List[int]:
        """
        Regression for parents variables selection.

        Implementation of parents selection for `child` node.
        Returns parents of node `child` associated to sample `y`.

        Parameters
        ----------
        X : np.ndarray
            exogenous variables.
        y : np.ndarray
            endogenous variables.
        pot_parents: List[int]
            List of potential parents admitted by the topological ordering.
        child : int
            Child node with `pot_parents` potential parent nodes.

        Returns
        -------
        parents : List[int]
            The list of selected parents for the input child node.
        """
        _, d = X.shape

        gam = self._fit_gam_model(X, y)
        pvalues = gam.statistics_["p_values"]

        parents = []
        for j in range(d):
            if pvalues[j] < self.cam_cutoff or self.context.included_edges.has_edge(
                pot_parents[j], child
            ):
                parents.append(pot_parents[j])
        return parents

    def _fit_gam_model(self, X: NDArray, y: NDArray) -> LinearGAM:
        """
        Fit GAM on `X` and `y`.

        Parameters
        ----------
        X : np.ndarray
            exogenous variables.
        y : np.ndarray
            endogenous variables.

        Returns
        -------
        gam : LinearGAM
            Fitted GAM with tuned lambda.

        Notes
        -----
        See https://pygam.readthedocs.io/en/latest/api/lineargam.html
        for implementation details of `LinearGAM`.
        """
        lambda_grid = {1: [0.1, 0.5, 1], 20: [5, 10, 20], 100: [50, 80, 100]}

        try:
            n, d = X.shape
        except ValueError:
            raise ValueError(
                (
                    f"not enough values to unpack (expected 2, got {len(X.shape)}). "
                    + "If vector X has only 1 dimension, try X.reshape(-1, 1)"
                )
            )

        n_splines = self._compute_n_splines(n, d)

        # Preliminary search of the best lambda between lambda_grid.keys()
        splines = [s(i, n_splines=n_splines, spline_order=self.degree) for i in range(d)]
        formula = self._make_formula(splines)
        lam_keys = lambda_grid.keys()
        gam = LinearGAM(formula, fit_intercept=False).gridsearch(
            X, y, lam=lam_keys, progress=False, objective="GCV"
        )
        lambdas = np.squeeze([s.get_params()["lam"] for s in gam.terms], axis=1)

        # Search the best lambdas according to the preliminary search, and get the fitted model
        lam = np.squeeze([lambda_grid[value] for value in lambdas])
        gam = LinearGAM(formula, fit_intercept=False).gridsearch(
            X, y, lam=lam.transpose(), progress=False, objective="GCV"
        )
        return gam

    def _compute_n_splines(self, n: int, d: int) -> int:
        """
        Compute the number of splines used for GAM fitting.

        During GAM fitting, decrease number of splines in case of small sample size.

        Parameters
        ----------
        n : int
            Number of samples in the datasets.
        d : int
            Number of nodes in the graph.

        Returns
        -------
        n_splines : int
            Updated number of splines for GAM fitting.
        """
        n_splines = self.n_splines
        if n / d < 3 * self.n_splines:
            n_splines = math.ceil(n / (3 * self.n_splines))
            print(
                (
                    f"Changed number of basis functions to {n_splines} in order to have"
                    + " enough samples per basis function"
                )
            )
            if n_splines <= self.degree:
                warnings.warn(
                    (
                        f"n_splines must be > spline_order. found: n_splines = {n_splines}"
                        + f" and spline_order = {self.degree}."
                        + f" n_splines set to {self.degree + 1}"
                    )
                )
                n_splines = self.degree + 1
        return n_splines

    def _make_formula(self, splines_list: List[Term]) -> TermList:
        """
        Make formula for PyGAM model from list of splines Term objects.

        The method defines a linear combination of the spline terms.

        Parameters
        ----------
        splines_list : List[Term]
            List of splines terms for the GAM formula.
            Example: [s(0), s(1), s(2)] where s is a B-spline Term from pyGAM.
            The equivalent R formula would be "s(0) + s(1) + s(2)", while the y target
            is provided directly at gam.fit() call.

        Returns
        -------
        terms : TermList
            Formula of the type requested by pyGAM Generalized Additive Models class.

        Notes
        -----
        See https://pygam.readthedocs.io/en/latest/dev-api/terms.html
        for details on `Term` and `TermsList` implementations.
        """
        terms = TermList()
        for spline in splines_list:
            terms += spline
        return terms
