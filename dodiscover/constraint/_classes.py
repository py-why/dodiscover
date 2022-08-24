import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.constraint.skeleton import LearnSkeleton, SkeletonMethods
from dodiscover.context import Context
from dodiscover.typing import Column

from .._protocol import EquivalenceClassProtocol


class BaseConstraintDiscovery:
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
    skeleton_method : SkeletonMethods
        The method to use for testing conditional independence. Must be one of
        ('neighbors', 'complete', 'neighbors_path'). See Notes for more details.
    max_path_length : int
        The maximum length of a path to consider when looking for possibly d-separating
        sets among two nodes. Only used if ``skeleton_method=pds``. Default is infinite.
    apply_orientations : bool
        Whether or not to apply orientation rules given the learned skeleton graph
        and separating set per pair of variables. If ``True`` (default), will
        apply orientation rules for specific algorithm.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    graph_ : EquivalenceClassProtocol
        The equivalence class of graphs discovered.
    separating_sets_ : dict
        The dictionary of separating sets, where it is a nested dictionary from
        the variable name to the variable it is being compared to the set of
        variables in the graph that separate the two.
    """

    graph_: Optional[EquivalenceClassProtocol]
    separating_sets_: Optional[Dict[Column, Dict[Column, List[Set[Column]]]]]

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_combinations: int = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        max_path_length: int = np.inf,
        apply_orientations: bool = True,
        **ci_estimator_kwargs,
    ):
        self.alpha = alpha
        self.ci_estimator = ci_estimator
        self.ci_estimator_kwargs = ci_estimator_kwargs
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

        # initialize the result properties we want to fit
        self.separating_sets_ = None
        self.graph_ = None

    def _initialize_sep_sets(
        self, init_graph: nx.Graph
    ) -> Dict[Column, Dict[Column, List[Set[Column]]]]:
        # keep track of separating sets
        sep_set: Dict[Column, Dict[Column, List[Set[Column]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # since we are not starting from a complete graph, find the separating sets
        for (node_i, node_j) in itertools.combinations(init_graph.nodes, 2):
            if not init_graph.has_edge(node_i, node_j):
                sep_set[node_i][node_j] = []
                sep_set[node_j][node_i] = []

        return sep_set

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClassProtocol:
        raise NotImplementedError(
            "All constraint discovery algorithms need to implement a function to convert "
            "the skeleton graph to a causal graph."
        )

    def orient_unshielded_triples(
        self,
        graph: EquivalenceClassProtocol,
        sep_set: Dict[Column, Dict[Column, List[Set[Column]]]],
    ) -> None:
        raise NotImplementedError()

    def orient_edges(self, graph: EquivalenceClassProtocol) -> None:
        raise NotImplementedError(
            "All constraint discovery algorithms need to implement a function to orient the "
            "skeleton graph given a separating set."
        )

    def fit(self, context: Context) -> None:
        """Fit constraint-based discovery algorithm on dataset 'X'.

        Parameters
        ----------
        X : Union[pd.DataFrame, Dict[Set, pd.DataFrame]]
            Either a pandas dataframe constituting the endogenous (observed) variables
            as columns and samples as rows, or a dictionary of different sampled
            distributions with keys as the distribution names and values as the dataset
            as a pandas dataframe.

        Raises
        ------
        RuntimeError
            If 'X' is a dictionary, then all datasets should have the same set of column
            names (nodes).

        Notes
        -----
        Control over the constraints imposed by the algorithm can be passed into the class
        constructor.
        """
        self.context_ = context
        graph = context.init_graph
        self.init_graph_ = graph
        self.fixed_edges_ = context.included_edges

        # create a reference to the underlying data to be used
        self.X_ = context.data

        # initialize graph object to apply learning
        sep_set = self._initialize_sep_sets(self.init_graph_)

        # learn skeleton graph and the separating sets per variable
        graph, sep_set = self.learn_skeleton(context, sep_set)

        # convert networkx.Graph to relevant causal graph object
        graph = self.convert_skeleton_graph(graph)

        # orient edges on the causal graph object
        if self.apply_orientations:
            # for all pairs of non-adjacent variables with a common neighbor
            # check if we can orient the edge as a collider
            self.orient_unshielded_triples(graph, sep_set)
            self.orient_edges(graph)

        # store resulting data structures
        self.separating_sets_ = sep_set
        self.graph_ = graph

    def evaluate_edge(
        self, data: pd.DataFrame, X: Column, Y: Column, Z: Optional[Set[Column]] = None
    ) -> Tuple[float, float]:
        """Test any specific edge for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``.
        Y : column
            A column in ``data``.
        Z : set, optional
            A list of columns in ``data``, by default None.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        if Z is None:
            Z = set()
        test_stat, pvalue = self.ci_estimator.test(data, {X}, {Y}, Z, **self.ci_estimator_kwargs)
        return test_stat, pvalue

    def learn_skeleton(
        self,
        context: Context,
        sep_set: Optional[Dict[Column, Dict[Column, List[Set[Column]]]]] = None,
    ) -> Tuple[nx.Graph, Dict[Column, Dict[Column, List[Set[Column]]]]]:
        """Learns the skeleton of a causal DAG using pairwise independence testing.

        Encodes the skeleton via an undirected graph, `networkx.Graph`. Only
        tests with adjacent nodes in the conditioning set.

        Parameters
        ----------
        context : Context
            A context object.
        sep_set : dict of dict of list of set
            The separating set.

        Returns
        -------
        skel_graph : nx.Graph
            The undirected graph of the causal graph's skeleton.
        sep_set : dict of dict of list of set
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
        skel_alg = LearnSkeleton(
            self.ci_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.skeleton_method,
            keep_sorted=False,
            **self.ci_estimator_kwargs,
        )
        skel_alg.fit(context)

        skel_graph = skel_alg.adj_graph_
        sep_set = skel_alg.sep_set_

        return skel_graph, sep_set
