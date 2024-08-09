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
from sklearn.metrics.pairwise import rbf_kernel

from dodiscover.context import Context
from dodiscover.toporder.utils import full_adj_to_order, full_dag, kernel_width

# -------------------- Mixin class with Stein estimators -------------------- #


class SteinMixin:
    """
    Implementation of Stein gradient estimator and Stein Hessian estimator and helper methods.
    """

    def hessian(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        """Stein estimator of the Hessian of log p(x).

        The Hessian matrix is efficiently estimated by exploitation of the Stein identity.
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
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s=s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)

        # Compute the Hessian by column stacked together
        H = np.stack([self._hessian_col(X, G, col, eta_H, K, s) for col in range(d)], axis=1)
        return H

    def score(
        self,
        X: NDArray,
        eta_G: float,
        K: Optional[NDArray] = None,
        nablaK: Optional[NDArray] = None,
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
            s = kernel_width(X)
            K = self._evaluate_kernel(X, s)
            nablaK = self._evaluate_nablaK(K, X, s)

        G = np.matmul(np.linalg.inv(K + eta_G * np.eye(n)), nablaK)
        return G

    def _hessian_col(
        self, X: NDArray, G: NDArray, c: int, eta: float, K: NDArray, s: float
    ) -> NDArray:
        """Stein estimator of a column of the Hessian of log p(x)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data
        G : np.ndarray
            estimator of the score function.
        c : int
            index of the column of interest.
        eta: float
            regularization parameter for ridge regression in Stein hessian estimator
        K : np.ndarray of shape (n_samples, n_samples)
            Gaussian kernel evaluated at X.
        s : float
            Width of the Gaussian kernel.

        Returns
        -------
        H_col : np.ndarray
            Stein estimator of the c-th column of the Hessian of log p(x)
        """
        X_diff = self._X_diff(X)
        n, _, _ = X_diff.shape

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

        # Score function compute
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)

        # Hessian compute
        X_diff = self._X_diff(X)
        nabla2K = np.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
        return -(G**2) + np.matmul(np.linalg.inv(K + eta_H * np.eye(n)), nabla2K)

    def _evaluate_kernel(self, X: NDArray, s: float) -> Tuple[NDArray, Union[NDArray, None]]:
        """
        Evaluate Gaussian kernel from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        s : float
            Width of the Gaussian kernel.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            Evaluated gaussian kernel.
        """
        K = rbf_kernel(X, gamma=1 / (2 * s**2)) / s
        return K

    def _evaluate_nablaK(self, K: NDArray, X: NDArray, s: float):
        """Evaluate <nabla, K> inner product.

        Parameters
        ----------
        K : np.ndarray of shape (n_samples, n_samples)
            Evaluated gaussian kernel of the matrix of the data.
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        s : float
            Width of the Gaussian kernel.

        Returns
        -------
        nablaK : np.ndarray of shape (n_samples, n_nodes)
            Inner product between the Gram matrix of the data and the kernel matrix K.
            To obtain the Gram matrix, for each sample of X, compute its difference
            with all n_samples.
        """
        nablaK = -np.einsum("kij,ik->kj", self._X_diff(X), K) / s**2
        return nablaK

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


# -------------------- Class with CAM-pruning implementation --------------------#


class CAMPruning:
    """Class implementing regression based selection of edges of a DAG.

    Implementation of the CAM-pruning method :footcite:`Buhlmann2013`.
    The algorithm performs selection of edges of an input DAG via hypothesis
    testing on the coefficients of a regression model.

    Parameters
    ----------
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.001.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10. Automatically decreased
        in case of insufficient samples.
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, alpha: float = 0.05, n_splines: int = 10, splines_degree: int = 3):
        self.alpha = alpha
        self.n_splines = n_splines
        self.degree = splines_degree

    def prune(
        self, X: NDArray, A_dense: NDArray, G_included: nx.DiGraph, G_excluded: nx.DiGraph
    ) -> NDArray:
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
        G_included : nx.DiGraph
            Graph with edges that are required to be included in the output.
            It encodes assumptions and prior knowledge about the causal graph.
        G_excluded : nx.DiGraph
            Graph with edges that are required to be excluded from the output.
            It encodes assumptions and prior knowledge about the causal graph.

        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        _, d = X.shape
        A = np.zeros((d, d))
        order = full_adj_to_order(A_dense)
        for c in range(d):
            pot_parents = []
            for p in order[: order.index(c)]:
                if ((not G_excluded.has_edge(p, c)) and A_dense[p, c] == 1) or G_included.has_edge(
                    p, c
                ):
                    pot_parents.append(p)
            if len(pot_parents) > 0:
                parents = self._variable_selection(
                    X[:, pot_parents], X[:, c], pot_parents, c, G_included
                )
                for parent in parents:
                    A[parent, c] = 1

        return A

    def _variable_selection(
        self,
        X: NDArray,
        y: NDArray,
        pot_parents: List[int],
        child: int,
        G_included: nx.DiGraph,
    ) -> List[int]:
        """
        Regression for parents variables selection.

        Implementation of parents selection for `child` node.
        Returns parents of node `child` associated to sample `y`.

        Parameters
        ----------
        X : np.ndarray
            Exogenous variables.
        y : np.ndarray
            Endogenous variables.
        pot_parents: List[int]
            List of potential parents admitted by the topological ordering.
        child : int
            Child node with `pot_parents` potential parent nodes.
        G_included : nx.DiGraph
            Graph with edges that are required to be included in the output.
            It encodes assumptions and prior knowledge about the causal graph.

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
            if pvalues[j] < self.alpha or G_included.has_edge(pot_parents[j], child):
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
            is provided directly at gam.learn_graph() call.

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


# -------------------- Topological ordering interface -------------------- #


class TopOrderInterface(metaclass=ABCMeta):
    """
    Interface for causal discovery based on estimate of topologial ordering and DAG pruning.
    """

    @abstractmethod
    def learn_graph(self, data: pd.DataFrame, context: Optional[Context] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        raise NotImplementedError()

    @abstractmethod
    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        raise NotImplementedError()


# -------------------- Base class for order-based inference --------------------#
class BaseTopOrder(CAMPruning, TopOrderInterface):
    """Base class for order-based causal discovery.

    Implementation of `TopOrderInterface` defining `fit` method for causal discovery.
    Class inheriting from `BaseTopOrder` need to implement the `top_order` method for inference
    of the topological ordering of the nodes in the causal graph.
    The resulting fully connected matrix is pruned by `prune` method implementation of
    CAM pruning :footcite:`Buhlmann2013`.

    Parameters
    ----------
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.001.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
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
    order_graph_ : nx.DiGraph
        Fully connected adjacency matrix representation of the
        inferred topological order.
    labels_to_nodes : Dict[Union[str, int], int]
        Map from the custom node's label  to the node's label by number.
    nodes_to_labels : Dict[int, Union[str, int]]
        Map from the node's label by number to the custom node's label.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        alpha: float = 0.05,
        prune: bool = True,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        # Initialize CAMPruning
        super().__init__(alpha, n_splines, splines_degree)

        # Parameters
        self.apply_pruning = prune
        self.do_pns = pns
        self.pns_num_neighbors = pns_num_neighbors
        self.pns_threshold = pns_threshold

        # Attributes
        self.graph_: nx.DiGraph = nx.empty_graph()
        self.order_: List[int] = list()
        self.order_graph_: nx.DiGraph = nx.empty_graph()
        self.labels_to_nodes: Dict[Union[str, int], int] = dict()
        self.nodes_to_labels: Dict[int, Union[str, int]] = dict()

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

    def learn_graph(self, data_df: pd.DataFrame, context: Optional[Context] = None) -> None:
        """
        Fit topological order based causal discovery algorithm on input data.

        Parameters
        ----------
        data_df : pd.DataFrame
            Datafame of the input data.
        context: Context
            The context of the causal discovery problem.
        """
        if context is None:
            # make a private Context object to store causal context used in this algorithm
            # store the context
            from dodiscover.context_builder import make_context

            context = make_context().build()

        X = data_df.to_numpy()
        self.context = context

        # Data structure to exchange labels with nodes number
        self.nodes_to_labels = {i: data_df.columns[i] for i in range(len(data_df.columns))}
        self.labels_to_nodes = {data_df.columns[i]: i for i in range(len(data_df.columns))}

        # Check acyclicity condition on included_edges
        self._dag_check_included_edges()
        self.order_constraints = self._included_edges_order_constraints()

        # Inference of the causal order.
        A_dense, order = self._top_order(X)
        self.order_ = order
        order_graph_ = nx.from_numpy_array(full_dag(order), create_using=nx.DiGraph)

        # Inference of the causal graph via pruning
        if self.apply_pruning:
            A = self._prune(X, A_dense)
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        else:
            G = nx.from_numpy_array(full_dag(order), create_using=nx.DiGraph)  # order_graph

        # Relabel the nodes according to the input data_df columns
        self.graph_ = self._postprocess_output(G)
        self.order_graph_ = self._postprocess_output(order_graph_)

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
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
        G_included = self._get_included_edges_graph()
        G_excluded = self._get_excluded_edges_graph()
        return super().prune(X, A_dense, G_included, G_excluded)

    def _dag_check_included_edges(self) -> None:
        """Check that the edges included in `self.context` does not violate DAG condition."""
        is_dag = nx.is_directed_acyclic_graph(self._get_included_edges_graph())
        if nx.is_empty(self._get_included_edges_graph()):
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
        adj = nx.to_numpy_array(self._get_included_edges_graph())
        d, _ = adj.shape
        descendants: Dict = {i: list() for i in range(d)}
        for row in range(d):
            for col in range(d):
                if adj[row, col] == 1:
                    row_descendants = descendants[row]
                    row_descendants.append(col)
                    descendants[row] = row_descendants
        return descendants

    def _postprocess_output(self, graph):
        """Relabel the graph nodes with the custom labels of the input dataframe.

        Parameters
        ----------
        graph : nx.DiGraph
            Networkx directed graph with nodes to relabel.

        Returns
        -------
        G : nx.DiGraph
            Graph with the relabeled nodes.
        """
        G = nx.relabel_nodes(graph, mapping=self.nodes_to_labels)
        return G

    def _get_included_edges_graph(self):
        """Get the `self.context.included_edges` graph with numerical label of the nodes.

        The returned directed graph of the included edges, with one node for each column
        in the dataframe input of the `fit` method, and numerical label of the nodes.

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph of the edges included in `self.context`.
        """
        num_nodes = len(self.labels_to_nodes)
        G = nx.empty_graph(n=num_nodes, create_using=nx.DiGraph)
        for edge in self.context.included_edges.edges():
            u, v = self.labels_to_nodes[edge[0]], self.labels_to_nodes[edge[1]]
            G.add_edge(u, v)
        return G

    def _get_excluded_edges_graph(self):
        """Get the `self.context.excluded_edges` graph with numerical label of the nodes.

        The returned directed graph of the excluded edges, with one node for each column
        in the dataframe input of the `fit` method, and numerical label of the nodes.

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph of the edges excluded in `self.context`.
        """
        num_nodes = len(self.labels_to_nodes)
        G = nx.empty_graph(n=num_nodes, create_using=nx.DiGraph)
        for edge in self.context.excluded_edges.edges():
            u, v = self.labels_to_nodes[edge[0]], self.labels_to_nodes[edge[1]]
            G.add_edge(u, v)
        return G
