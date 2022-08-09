import itertools
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from pywhy_graphs import ADMG, PAG
from ._protocol import GraphProtocol, EquivalenceClassProtocol
from dodiscover.ci.base import BaseConditionalIndependenceTest
from graphs import MixedEdgeGraph

# TODO: Add ways to fix directed edges
# TODO: Add ways to initialize graph with edges rather then undirected
class ConstraintDiscovery:
    """Constraint-based algorithms for causal discovery.

    Contains common methods used for all constraint-based causal discovery algorithms.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    init_graph : nx.Graph | ADMG, optional
        An initialized graph. If ``None``, then will initialize PC using a
        complete graph. By default None.
    fixed_edges : nx.Graph, optional
        An undirected graph with fixed edges. If ``None``, then will initialize PC using a
        complete graph. By default None.
    min_cond_set_size : int, optional
        Minimum size of the conditioning set, by default None, which will be set to '0'.
        Used to constrain the computation spent on the algorithm.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        Maximum number of tries with a conditioning set chosen from the set of possible
        parents still, by default None. If None, then will not be used. If set, then
        the conditioning set will be chosen lexographically based on the sorted
        test statistic values of 'ith Pa(X) -> X', for each possible parent node of 'X'.
    skeleton_method : str
        The method to use for testing conditional independence. Must be one of
        ('neighbors', 'complete'). See Notes for more details.
    max_path_length : int
        The maximum length of a path to consider when looking for possibly d-separating
        sets among two nodes. Only used if ``skeleton_method=pds``. Default is infinite.
    pag : PAG
        The partial ancestral graph. Only used if ``skeleton_method=pds``.
    apply_orientations : bool
        Whether or not to apply orientation rules given the learned skeleton graph
        and separating set per pair of variables. If ``True`` (default), will
        apply orientation rules for specific algorithm.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    graph_ : PAG
        The graph discovered.
    separating_sets_ : dict
        The dictionary of separating sets, where it is a nested dictionary from
        the variable name to the variable it is being compared to the set of
        variables in the graph that separate the two.
    """

    graph_: Optional[Any]
    separating_sets_: Optional[Dict[str, Dict[str, Set[Any]]]]

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph,MixedEdgeGraph] = None,
        fixed_edges: Union[nx.Graph,MixedEdgeGraph] = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_combinations: int = None,
        skeleton_method: str = "neighbors",
        max_path_length: int = np.inf,
        pag: EquivalenceClassProtocol = None,
        apply_orientations: bool = True,
        **ci_estimator_kwargs,
    ):
        self.alpha = alpha
        self.ci_estimator = ci_estimator
        self.ci_estimator_kwargs = ci_estimator_kwargs
        self.init_graph = init_graph
        self.fixed_edges = fixed_edges
        self.apply_orientations = apply_orientations
        self.skeleton_method = skeleton_method

        # constraining the conditional independence tests
        if max_cond_set_size is None:
            max_cond_set_size = np.inf
        self.max_cond_set_size = max_cond_set_size
        if min_cond_set_size is None:
            min_cond_set_size = 0
        self.min_cond_set_size = min_cond_set_size
        if max_combinations is None:
            max_combinations = np.inf
        self.max_combinations = max_combinations

        # special attributes for learning skeleton with semi-Markovian models
        self.max_path_length = max_path_length
        self.pag = pag

        # initialize the result properties we want to fit
        self.separating_sets_ = None
        self.graph_ = None

    def _initialize_fixed_constraints(self, nodes):
        # check on fixed edges and keep track
        fixed_edges = set()
        if self.fixed_edges is not None:
            if not np.array_equal(self.fixed_edges.nodes, nodes):
                raise ValueError(
                    f"The nodes within the fixed-edges graph, {self.fixed_edges.nodes}, "
                    f"do not match the nodes in the passed in data, {nodes}."
                )

            for (i, j) in self.fixed_edges.edges:
                fixed_edges.add((i, j))
                fixed_edges.add((j, i))
        return fixed_edges

    def _initialize_graph(self, nodes):
        # keep track of separating sets
        sep_set: Dict[str, Dict[str, Set]] = defaultdict(lambda: defaultdict(list))

        # initialize the starting graph
        if self.init_graph is None:
            graph = nx.complete_graph(nodes, create_using=nx.Graph)
        else:
            graph = self.init_graph

            if graph.nodes != nodes:
                raise ValueError(
                    f"The nodes within the initial graph, {graph.nodes}, "
                    f"do not match the nodes in the passed in data, {nodes}."
                )

            # since we are not starting from a complete graph,
            # find the separating sets
            for (node_i, node_j) in itertools.combinations(*graph.nodes):
                if not graph.has_edge(node_i, node_j):
                    sep_set[node_i][node_j] = []
                    sep_set[node_j][node_i] = []

        return graph, sep_set

    def orient_edges(self, graph: GraphProtocol, sep_set: Dict[str, Dict[str, Set]]) -> Any:
        raise NotImplementedError(
            "All constraint discovery algorithms need to implement a function to orient the "
            "skeleton graph given a separating set."
        )

    def _orient_unshielded_triples(self, graph: GraphProtocol, sep_set: Dict[str, Dict[str, Set]]):
        raise NotImplementedError()

    def convert_skeleton_graph(self, graph: nx.Graph) -> GraphProtocol:
        raise NotImplementedError(
            "All constraint discovery algorithms need to implement a function to convert "
            "the skeleton graph to a causal graph."
        )

    def fit(self, X: Union[pd.DataFrame, Dict[Set, pd.DataFrame]]) -> None:
        """Fit constraint-based discovery algorithm on dataset 'X'.

        Parameters
        ----------
        X : Union[pd.DataFrame, Dict[Set, pd.DataFrame]]
            Either a pandas dataframe constituting the endogenous (observed) variables
            as columns and samples as rows, or a dictionary of different sampled distributions
            with keys as the distribution names and values as the dataset as a pandas dataframe.

        Raises
        ------
        RuntimeError
            If 'X' is a dictionary, then all datasets should have the same set of column names (nodes).

        Notes
        -----
        Control over the constraints imposed by the algorithm can be passed into the class constructor.
        """
        # perform error-checking and extract node names
        if isinstance(X, dict):
            # the data passed in are instances of multiple distributions
            for idx, (_, X_dataset) in enumerate(X.items()):
                if idx == 0:
                    check_nodes = X_dataset.columns
                nodes = X_dataset.columns
                if not check_nodes.equals(nodes):
                    raise RuntimeError(
                        "All dataset distributions should have the same node names in their columns."
                    )

            # convert final series of nodes to a list
            nodes = nodes.values
        else:
            nodes = X.columns.values

        # create a reference to the underlying data to be used
        self.X = X

        # initialize graph object to apply learning
        graph, sep_set = self._initialize_graph(nodes)

        # initialize fixed edge constraints
        fixed_edges = self._initialize_fixed_constraints(nodes)

        # learn skeleton graph and the separating sets per variable
        graph, sep_set, _, _ = self.learn_skeleton(X, graph, sep_set, fixed_edges)

        # convert networkx.Graph to relevant causal graph object
        graph = self.convert_skeleton_graph(graph)

        # orient edges on the causal graph object
        if self.apply_orientations:
            graph = self.orient_edges(graph, sep_set)

        # store resulting data structures
        self.separating_sets_ = sep_set
        self.graph_ = graph

    def test_edge(self, data, X, Y, Z=None):
        """Test any specific edge for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``.
        Y : column
            A column in ``data``.
        Z : list, optional
            A list of columns in ``data``, by default None.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        if Z is None:
            Z = []
        test_stat, pvalue = self.ci_estimator.test(data, X, Y, set(Z), **self.ci_estimator_kwargs)
        return test_stat, pvalue

    def learn_skeleton(
        self,
        X: pd.DataFrame,
        graph: Optional[nx.Graph] = None,
        sep_set: Optional[Dict[str, Dict[str, Set[Any]]]] = None,
        fixed_edges: Optional[Set] = None,
    ) -> Tuple[
        nx.Graph,
        Dict[str, Dict[str, Set[Any]]],
        Dict[Any, Dict[Any, float]],
        Dict[Any, Dict[Any, float]],
    ]:
        """Learns the skeleton of a causal DAG using pairwise independence testing.

        Encodes the skeleton via an undirected graph, `networkx.Graph`. Only
        tests with adjacent nodes in the conditioning set.

        Parameters
        ----------
        X : pd.DataFrame
            The data with columns as variables and samples as rows.
        graph : nx.Graph
            The undirected graph containing initialized skeleton of the causal
            relationships.
        sep_set : set
            The separating set.
        fixed_edges : set, optional
            The set of fixed edges. By default, is the empty set.
        return_deps : bool
            Whether to return the two mappings for the dictionary of test statistic
            and pvalues.

        Returns
        -------
        skel_graph : nx.Graph
            The undirected graph of the causal graph's skeleton.
        sep_set : dict of dict of set
            The separating set per pairs of variables.

        Raises
        ------
        ValueError
            If the nodes in the initialization graph do not match the variable
            names in passed in data, ``X``.
        ValueError
            If the nodes in the fixed-edge graph do not match the variable
            names in passed in data, ``X``.

        Notes
        -----
        Learning the skeleton of a causal DAG uses (conditional) independence testing
        to determine which variables are (in)dependent. This specific algorithm
        compares exhaustively pairs of adjacent variables.
        """
        from causal_networkx.discovery.skeleton import LearnSkeleton

        if fixed_edges is None:
            fixed_edges = set()

        skel_alg = LearnSkeleton(
            self.ci_estimator,
            adj_graph=graph,
            sep_set=sep_set,
            fixed_edges=fixed_edges,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.skeleton_method,
            # max_path_length=self.max_path_length,
            # pag=self.pag,
            keep_sorted=False,
            **self.ci_estimator_kwargs,
        )
        skel_alg.fit(X)

        skel_graph = skel_alg.adj_graph_
        sep_set = skel_alg.sep_set_
        test_stat_dict = skel_alg.test_stat_dict_
        pvalue_dict = skel_alg.pvalue_dict_

        return skel_graph, sep_set, test_stat_dict, pvalue_dict
