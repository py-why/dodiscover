import logging
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from itertools import chain, combinations
from typing import Iterable, List, Optional, Set, SupportsFloat, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover.cd import BaseConditionalDiscrepancyTest
from dodiscover.ci import BaseConditionalIndependenceTest, Oracle
from dodiscover.constraint.config import SkeletonMethods
from dodiscover.constraint.utils import is_in_sep_set
from dodiscover.typing import Column, SeparatingSet

from .._protocol import EquivalenceClass
from ..context import Context
from ..context_builder import ContextBuilder, InterventionalContextBuilder, make_context

logger = logging.getLogger()


def _iter_conditioning_set(
    possible_variables: Iterable,
    x_var: Union[SupportsFloat, str],
    y_var: Union[SupportsFloat, str],
    size_cond_set: int,
) -> Iterable[Set]:
    """Iterate function to generate the conditioning set.

    Parameters
    ----------
    possible_variables : iterable
        A set/list/dict of possible variables to consider for the conditioning set.
        This can be for example, the current adjacencies.
    x_var : node
        The node for the 'x' variable.
    y_var : node
        The node for the 'y' variable.
    size_cond_set : int
        The size of the conditioning set to consider. If there are
        less adjacent variables than this number, then all variables will be in the
        conditioning set.

    Yields
    ------
    Z : set
        The set of variables for the conditioning set.
    """
    exclusion_set = {x_var, y_var}

    all_adj_excl_current = [p for p in possible_variables if p not in exclusion_set]

    # loop through all possible combinations of the conditioning set size
    for cond in combinations(all_adj_excl_current, size_cond_set):
        cond_set = set(cond)
        yield cond_set


def _find_neighbors_along_path(G: nx.Graph, start, end) -> Set:
    """Find neighbors that are along a path from start to end.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    start : Node
        The starting node.
    end : Node
        The ending node.

    Returns
    -------
    nbrs : Set
        The set of neighbors that are also along a path towards
        the 'end' node.
    """
    nbrs = set()

    # query all neighbors of X and then only add nodes that are in a valid path
    # to end
    for node in G.neighbors(start):
        if not G.has_edge(start, node):
            raise RuntimeError(f"{start} and {node} are not connected, but they are assumed to be.")

        # if we queried the edge we are testing, then pick that one
        if node == end:
            continue

        # find a path from start node to end
        paths = nx.all_simple_paths(G, source=node, target=end)
        for path in paths:
            # the trivial path which indicates that 'node' is only connected to
            # 'end' through 'start'
            if path == (node, start, end):
                continue
            else:
                # found a single path
                nbrs.add(node)
                break
    return nbrs


class LearnSkeleton:
    """Learn a skeleton graph from observational data without latent confounding.

    A skeleton graph from a Markovian causal model can be learned completely
    with this procedure.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    skeleton_method : SkeletonMethods
        The method to use for testing conditional independence. Must be one of
        ('complete', 'neighbors', 'neighbors_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph. The graph contains edge attributes for the smallest value of the
        test statistic encountered (key name 'test_stat'), the largest pvalue seen in
        testing 'x' || 'y' given some conditioning set (key name 'pvalue').
    sep_set_ : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
    context_ : Context
        The result context. Encodes causal assumptions.
    min_cond_set_size_ : int
        The inferred minimum conditioning set size.
    max_cond_set_size_ : int
        The inferred maximum conditioning set size.
    max_combinations_ : int
        The inferred maximum number of combinations of 'Z' to test per
        :math:`X \\perp Y | Z`.
    n_iters_ : int
        The number of iterations the skeleton has been learned.

    Notes
    -----
    Proceed by testing neighboring nodes, while keeping track of test
    statistic values (these are the ones that are
    the "most dependent"). Remember we are testing the null hypothesis

    .. math::
        H_0: X \\perp Y | Z

    where the alternative hypothesis is that they are dependent and hence
    require a causal edge linking the two variables.

    Overview of learning causal skeleton from data:

        This algorithm consists of four general loops through the data.

        - "infinite" loop through size of the conditioning set, 'size_cond_set'. The
          minimum size is set by ``min_cond_set_size``, whereas the maximum is controlled
          by ``max_cond_set_size`` hyperparameter.
        - loop through nodes of the graph, 'x_var'
        - loop through variables adjacent to selected node, 'y_var'
        - loop through combinations of the conditioning set of size p, 'cond_set'.
          The ``max_combinations`` parameter allows one to limit the fourth loop through
          combinations of the conditioning set.

        At each iteration of the outer infinite loop, the edges that were deemed
        independent for a specific 'size_cond_set' are removed and 'size_cond_set' is incremented.

        Furthermore, the maximum pvalue is stored for existing
        dependencies among variables (i.e. any two nodes with an edge still).
        The ``keep_sorted`` hyperparameter keeps the considered neighbors in
        a sorted order.

        The stopping condition is when the size of the conditioning variables for all (X, Y)
        pairs is less than the size of 'size_cond_set', or if the 'max_cond_set_size' is
        reached.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``skeleton_method`` completely defines how one
        selects the conditioning set.

        - 'complete': This exhaustively conditions on all combinations of variables in
          the graph. This essentially refers to the SGS algorithm :footcite:`Spirtes1993`
        - 'neighbors': This only conditions on adjacent variables to that of 'x_var' and 'y_var'.
          This refers to the traditional PC algorithm :footcite:`Meek1995`
        - 'neighbors_path': This is 'neighbors', but restricts to variables with an adjacency path
          from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`
    """

    adj_graph_: nx.Graph
    context_: Context
    min_cond_set_size_: int
    max_cond_set_size_: int
    max_combinations_: int
    sep_set_: SeparatingSet
    n_iters_: int = 0

    remove_edges: Set

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        keep_sorted: bool = False,
        ci_estimator_kwargs=None,
    ) -> None:
        self.ci_estimator = ci_estimator
        self.sep_set = sep_set
        self.alpha = alpha
        self.ci_estimator_kwargs = ci_estimator_kwargs
        self.skeleton_method = skeleton_method

        # control of the conditioning set
        self.min_cond_set_size = min_cond_set_size
        self.max_cond_set_size = max_cond_set_size
        self.max_combinations = max_combinations

        # for tracking strength of dependencies
        self.keep_sorted = keep_sorted

        # debugging mode
        self.n_ci_tests = 0

    def _initialize_params(self) -> None:
        """Initialize parameters for learning skeleton.

        Basic parameters that are used by any constraint-based causal discovery algorithms.
        """
        # error checks of passed in arguments
        if self.max_combinations is not None and self.max_combinations <= 0:
            raise RuntimeError(f"Max combinations must be at least 1, not {self.max_combinations}")

        if self.skeleton_method not in SkeletonMethods:
            raise ValueError(
                f"Skeleton method must be one of {SkeletonMethods}, not {self.skeleton_method}."
            )

        if self.sep_set is None and not hasattr(self, "sep_set_"):
            # keep track of separating sets
            self.sep_set_ = defaultdict(lambda: defaultdict(list))
        elif not hasattr(self, "sep_set_"):
            self.sep_set_ = self.sep_set  # type: ignore

        # control of the conditioning set
        if self.max_cond_set_size is None:
            self.max_cond_set_size_ = np.inf
        else:
            self.max_cond_set_size_ = self.max_cond_set_size
        if self.min_cond_set_size is None:
            self.min_cond_set_size_ = 0
        else:
            self.min_cond_set_size_ = self.min_cond_set_size
        if self.max_combinations is None:
            self.max_combinations_ = np.inf
        else:
            self.max_combinations_ = self.max_combinations

        if self.ci_estimator_kwargs is None:
            self.ci_estimator_kwargs = dict()

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
        self.n_ci_tests += 1
        return test_stat, pvalue

    def fit(self, data: pd.DataFrame, context: Context):
        """Run structure learning to learn the skeleton of the causal graph.

        Parameters
        ----------
        data : pd.DataFrame
            The data to learn the causal graph from.
        context : Context
            A context object.
        """
        self.context_ = context.copy()

        # get the initialized graph
        adj_graph = deepcopy(self.context_.init_graph.copy())

        # track progress of the algorithm for which edges to remove to ensure stability
        self.remove_edges = set()

        # initialize learning parameters
        self._initialize_params()

        # the size of the conditioning set will start off at the minimum
        size_cond_set = self.min_cond_set_size_

        # allow us to query the iteration stage of the causal discovery algorithm
        # allowing us to run multiple iterations of the skeleton discovery
        edge_attrs = set(chain.from_iterable(d.keys() for *_, d in adj_graph.edges(data=True)))
        if self.n_iters_ == 0 and "test_stat" in edge_attrs or "pvalue" in edge_attrs:
            raise RuntimeError(
                "Running skeleton discovery with adjacency graph "
                "with 'test_stat' or 'pvalue' is not supported yet."
            )

        # store the absolute value of test-statistic values and pvalue for
        # every single candidate parent-child edge (X -> Y)
        nx.set_edge_attributes(adj_graph, np.inf, "test_stat")
        nx.set_edge_attributes(adj_graph, -1e-5, "pvalue")

        logger.info(
            f"\n\nRunning skeleton phase with: \n"
            f"max_combinations: {self.max_combinations_},\n"
            f"min_cond_set_size: {self.min_cond_set_size_},\n"
            f"max_cond_set_size: {self.max_cond_set_size_},\n"
        )

        # Outer loop: iterate over 'size_cond_set' until stopping criterion is met
        # - 'size_cond_set' > 'max_cond_set_size' or
        # - All (X, Y) pairs have candidate conditioning sets of size < 'size_cond_set'
        while 1:
            cont = False
            # initialize set of edges to remove at the end of every loop
            self.remove_edges = set()

            # loop through every node
            for x_var in adj_graph.nodes:
                possible_adjacencies = set(adj_graph.neighbors(x_var))

                logger.info(f"Considering node {x_var}...\n\n")

                for y_var in possible_adjacencies:
                    # a node cannot be a parent to itself in DAGs
                    if y_var == x_var:
                        continue

                    # ignore fixed edges
                    if (x_var, y_var) in self.context_.included_edges.edges:
                        continue

                    # compute the possible variables used in the conditioning set
                    possible_variables = self._compute_candidate_conditioning_sets(
                        adj_graph,
                        x_var,
                        y_var,
                    )

                    logger.debug(
                        f"Adj({x_var}) without {y_var} with size={len(possible_adjacencies) - 1} "
                        f"with p={size_cond_set}. The possible variables to condition on are: "
                        f"{possible_variables}."
                    )

                    # check that number of adjacencies is greater then the
                    # cardinality of the conditioning set
                    if len(possible_variables) < size_cond_set:
                        logger.debug(
                            f"\n\nBreaking for {x_var}, {y_var}, {len(possible_adjacencies)}, "
                            f"{size_cond_set}, {possible_variables}"
                        )
                        continue
                    else:
                        cont = True

                    # generate iterator through the conditioning sets
                    conditioning_sets = _iter_conditioning_set(
                        possible_variables=possible_variables,
                        x_var=x_var,
                        y_var=y_var,
                        size_cond_set=size_cond_set,
                    )

                    # now iterate through the possible parents
                    for comb_idx, cond_set in enumerate(conditioning_sets):
                        # check the number of combinations of possible parents we have tried
                        # to use as a separating set
                        if (
                            self.max_combinations_ is not None
                            and comb_idx >= self.max_combinations_
                        ):
                            break

                        # compute conditional independence test
                        test_stat, pvalue = self.evaluate_edge(data, x_var, y_var, set(cond_set))

                        # if any "independence" is found through inability to reject
                        # the null hypothesis, then we will break the loop comparing X and Y
                        # and say X and Y are conditionally independent given 'cond_set'
                        if pvalue > self.alpha:
                            break

                    # post-process the CI test results
                    removed_edge = self._postprocess_ci_test(
                        adj_graph, x_var, y_var, cond_set, test_stat, pvalue
                    )

                    # summarize the comparison of XY
                    self._summarize_xy_comparison(x_var, y_var, removed_edge, pvalue)

            # finally remove edges after performing
            # conditional independence tests
            logger.info(f"For p = {size_cond_set}, removing all edges: {self.remove_edges}")

            # Remove non-significant links
            # Note: Removing edges at the end ensures "stability" of the algorithm
            # with respect to the randomness choice of pairs of edges considered in the inner loop
            adj_graph.remove_edges_from(self.remove_edges)

            # increment the conditioning set size
            size_cond_set += 1

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > self.max_cond_set_size_ or cont is False:
                break

        self.adj_graph_ = adj_graph
        self.n_iters_ += 1

    def _summarize_xy_comparison(
        self, x_var: Column, y_var: Column, removed_edge: bool, pvalue: float
    ) -> None:
        # exit loop if we have found an independency and removed the edge
        if removed_edge:
            remove_edge_str = "Removing edge"
        else:
            remove_edge_str = "Did not remove edge"

        logger.info(
            f"{remove_edge_str} between {x_var} and {y_var}... \n"
            f"Statistical summary:\n"
            f"- PValue={pvalue} at alpha={self.alpha}"
        )

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        r"""Compute candidate conditioning sets.

        For a given 'X' and 'Y', this method implements a graphical algorithm that
        enumerates possible variables that are part of 'Z', the conditioning set.
        One can then test the following null hypothesis :math:`H_0: X \perp Y | Z`.

        Parameters
        ----------
        adj_graph : nx.Graph
            The current adjacency graph.
        x_var : node
            The 'X' node.
        y_var : node
            The 'Y' node.

        Returns
        -------
        possible_variables : Set of Column
            The set of nodes in 'adj_graph' that are candidates for the
            conditioning set.

        Notes
        -----
        The :attr:`skeleton_method` dictates how we choose the corresponding conditioning sets.
        For more information, see :class:`SkeletonMethods`.
        """
        skeleton_method = self.skeleton_method

        if skeleton_method == SkeletonMethods.COMPLETE:
            possible_variables = set(adj_graph.nodes)
        elif skeleton_method == SkeletonMethods.NBRS:
            possible_variables = set(adj_graph.neighbors(x_var))
            # possible_adjacencies.copy()
        elif skeleton_method == SkeletonMethods.NBRS_PATH:
            # constrain adjacency set to ones with a path from x_var to y_var
            possible_variables = _find_neighbors_along_path(adj_graph, start=x_var, end=y_var)

        if self.keep_sorted:
            # Note it is assumed in public API that 'test_stat' is set
            # inside the adj_graph
            possible_variables = sorted(
                possible_variables,
                key=lambda n: adj_graph.edges[x_var, n]["test_stat"],
                reverse=True,
            )  # type: ignore

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def _postprocess_ci_test(
        self,
        adj_graph: nx.Graph,
        x_var: Column,
        y_var: Column,
        cond_set: Set[Column],
        test_stat: float,
        pvalue: float,
    ) -> bool:
        # keep track of the smallest test statistic, meaning the highest pvalue
        # meaning the "most" independent. keep track of the maximum pvalue as well
        if pvalue > adj_graph.edges[x_var, y_var]["pvalue"]:
            adj_graph.edges[x_var, y_var]["pvalue"] = pvalue
        if test_stat < adj_graph.edges[x_var, y_var]["test_stat"]:
            adj_graph.edges[x_var, y_var]["test_stat"] = test_stat

        # two variables found to be independent given a separating set
        if pvalue > self.alpha:
            self.remove_edges.add((x_var, y_var))
            self.sep_set_[x_var][y_var].append(set(cond_set))
            self.sep_set_[y_var][x_var].append(set(cond_set))
            return True
        return False


class LearnSemiMarkovianSkeleton(LearnSkeleton):
    """Learning a skeleton from a semi-markovian causal model.

    This proceeds by learning a skeleton by testing edges with candidate
    separating sets from the "possibly d-separating" sets (PDS), or PDS
    sets that lie on a path between two nodes :footcite:`Spirtes1993`.
    This algorithm requires the input of a collider-oriented PAG, which
    provides the necessary information to compute the PDS set for any
    given nodes. See Notes for more details.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    skeleton_method : SkeletonMethods
        The method to use for determining conditioning sets when testing conditional
        independence of the first stage. See :class:`LearnSkeleton` for details.
    second_stage_skeleton_method : SkeletonMethods | None
        The method to use for determining conditioning sets when testing conditional
        independence of the first stage. Must be one of ('pds', 'pds_path'). See Notes
        for more details. If `None`, then no second stage skeleton discovery phase will be run.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph. The graph contains edge attributes for the smallest value of the
        test statistic encountered (key name 'test_stat'), the largest pvalue seen in
        testing 'x' || 'y' given some conditioning set (key name 'pvalue').
    sep_set_ : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
    context_ : Context
        The result context. Encodes causal assumptions.
    min_cond_set_size_ : int
        The inferred minimum conditioning set size.
    max_cond_set_size_ : int
        The inferred maximum conditioning set size.
    max_combinations_ : int
        The inferred maximum number of combinations of 'Z' to test per
        :math:`X \\perp Y | Z`.
    n_iters_ : int
        The number of iterations the skeleton has been learned.
    max_path_length_ : int
        Th inferred maximum path length any single discriminating path is allowed to take.

    Notes
    -----
    To learn the skeleton of a Semi-Markovian causal model, one approach is to consider
    the possibly d-separating (PDS) set, which is a superset of the d-separating sets in
    the true causal model. Knowing the PDS set requires knowledge of the skeleton and orientation
    of certain edges. Therefore, we first learn an initial skeleton by checking conditional
    independences with respect to node neighbors. From this, one can orient certain colliders.
    The resulting PAG can now be used to enumerate the PDS sets for each node, which
    are now conditioning candidates to check for conditional independence.

    For visual examples, see Figures 16, 17 and 18 in :footcite:`Spirtes1993`. Also,
    see the RFCI paper for other examples :footcite:`Colombo2012`.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``skeleton_method`` completely defines how one
        selects the conditioning set.

        - 'pds': This conditions on the PDS set of 'x_var'. Note, this definition does
          not rely on 'y_var'. See :footcite:`Spirtes1993`.
        - 'pds_path': This is 'pds', but restricts to variables with a possibly directed path
          from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`.

    References
    ----------
    .. footbibliography::
    """

    max_path_length_: int

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        second_stage_skeleton_method: Optional[SkeletonMethods] = SkeletonMethods.PDS,
        keep_sorted: bool = False,
        max_path_length: Optional[int] = None,
        ci_estimator_kwargs=None,
    ) -> None:
        super().__init__(
            ci_estimator,
            sep_set,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            keep_sorted,
            ci_estimator_kwargs=ci_estimator_kwargs,
        )

        self.second_stage_skeleton_method = second_stage_skeleton_method
        self.max_path_length = max_path_length

    def _orient_unshielded_triples(self, graph: EquivalenceClass, sep_set: SeparatingSet) -> None:
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : EquivalenceClass
            The partial ancestral graph (PAG).
        sep_set : SeparatingSet
            The separating set between any two nodes.
        """
        # for every node in the PAG, evaluate neighbors that have any edge
        for u in graph.nodes:
            for v_i, v_j in combinations(graph.neighbors(u), 2):
                # Check that there is no edge of any type between
                # v_i and v_j, else this is a "shielded" collider.
                # Then check to see if 'u' is in the separating
                # set. If it is not, then there is a collider.
                if v_j not in graph.neighbors(v_i) and not is_in_sep_set(
                    u, sep_set, v_i, v_j, mode="any"
                ):
                    if graph.has_edge(v_i, u, graph.circle_edge_name):
                        graph.orient_uncertain_edge(v_i, u)
                    if graph.has_edge(v_j, u, graph.circle_edge_name):
                        graph.orient_uncertain_edge(v_j, u)

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        import pywhy_graphs as pgraph

        # get PAG from the context object
        pag = self.context_.state_variable("PAG", on_missing="ignore")

        if pag is None:
            # if PAG has not been set as a state variable, then we are learning a skeleton
            # without PDS information. I.e. the normal LearnSkeleton algorithm
            return super()._compute_candidate_conditioning_sets(adj_graph, x_var, y_var)
        else:
            if not all(x in pag.nodes for x in [x_var, y_var]):
                raise RuntimeError("wtf..")
            skeleton_method = self.second_stage_skeleton_method

            if skeleton_method == SkeletonMethods.PDS:
                # determine how we want to construct the candidates for separating nodes
                # perform conditioning independence testing on all combinations
                possible_variables = pgraph.pds(
                    pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
                )
            elif skeleton_method == SkeletonMethods.PDS_PATH:
                # determine how we want to construct the candidates for separating nodes
                # perform conditioning independence testing on all combinations
                possible_variables = pgraph.pds_path(
                    pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
                )

            if self.keep_sorted:
                # Note it is assumed in public API that 'test_stat' is set
                # inside the adj_graph
                possible_variables = sorted(
                    possible_variables,
                    key=lambda n: adj_graph.edges[x_var, n]["test_stat"],
                    reverse=True,
                )  # type: ignore

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def fit(self, data: pd.DataFrame, context: Context):
        import pywhy_graphs as pgraphs

        if self.max_path_length is None:
            self.max_path_length_ = np.inf
        else:
            self.max_path_length_ = self.max_path_length

        # initially learn the skeleton without using PDS information
        super().fit(data, context)

        # if there is no second stage skeleton method to be run, then we
        # will stop with the skeleton here
        if self.second_stage_skeleton_method is None:
            return self

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        sep_set = self.sep_set_
        skel_graph = self.adj_graph_
        pag = pgraphs.PAG(incoming_circle_edges=skel_graph, name="PAG derived with FCI")

        # orient colliders
        self._orient_unshielded_triples(pag, sep_set)

        # convert the adjacency graph
        new_init_graph = pag.to_undirected()

        # Update the Context:
        # add the corresponding intermediate PAG now to the context
        # new initialization graph
        for (_, _, d) in new_init_graph.edges(data=True):
            if "test_stat" in d:
                d.pop("test_stat")
            if "pvalue" in d:
                d.pop("pvalue")
        context = (
            make_context(context).init_graph(new_init_graph).state_variable("PAG", pag).build()
        )

        if not all(x in context.state_variable("PAG").nodes for x in data.columns):
            raise RuntimeError("wtf..")

        # now compute all possibly d-separating sets and learn a better skeleton
        super().fit(data, context)
        return self


class LearnInterventionSkeleton(LearnSemiMarkovianSkeleton):
    """Learn skeleton using observational and interventional data.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    cd_estimator : BaseConditionalDiscrepancyTest
        The conditional discrepancy test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    skeleton_method : SkeletonMethods
        The method to use for testing conditional independence. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.
    cd_estimator_kwargs : dict
        Keyword arguments for the ``cd_estimator`` function.

    Notes
    -----
    With interventional data, one may either know the interventional targets from each
    experimental distribution dataset, or one may not know the explicit targets. If the
    interventional targets are known, then the skeleton discovery algorithm of
    :footcite:`Kocaoglu2019characterization` is used. That is we learn the skeleton of a
    IPAG. Otherwise, we will not know the intervention targets, and use the skeleton discovery
    algorithm described in :footcite:`Jaber2020causal`. To define intervention targets, one
    must use the :class:`dodiscover.InterventionalContextBuilder`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        cd_estimator: BaseConditionalDiscrepancyTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        second_stage_skeleton_method: SkeletonMethods = SkeletonMethods.PDS,
        keep_sorted: bool = False,
        max_path_length: Optional[int] = None,
        known_intervention_targets: bool = False,
        ci_estimator_kwargs=None,
        cd_estimator_kwargs=None,
    ) -> None:
        super().__init__(
            ci_estimator,
            sep_set,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            second_stage_skeleton_method,
            keep_sorted,
            max_path_length,
            ci_estimator_kwargs,
        )

        self.cd_estimator = cd_estimator
        self.known_intervention_targets = known_intervention_targets
        self.cd_estimator_kwargs = cd_estimator_kwargs

    def evaluate_fnode_edge(
        self,
        data: List[pd.DataFrame],
        X: Set[Column],
        Y: Set[Column],
        Z: Optional[Set[Column]] = None,
    ) -> Tuple[float, float]:
        """Test an edge from an F-node to a regular node for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``. This is assumed to be the F-node.
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

        # extract the F-node name
        group_col: Column = reduce(lambda x: x, X)  # type: ignore

        # get the sigma-map for this F-node
        distribution_idx = self.context_.sigma_map[group_col]

        # get the distributions across the two distributions
        data_i = data[distribution_idx[0]]
        data_j = data[distribution_idx[1]]

        # name the group column the F-node, so Oracle works as expected
        data_i[group_col] = 0
        data_j[group_col] = 1
        data = pd.concat((data_i, data_j), axis=0)

        # compare conditional distributions P(Y | X) vs P'(Y | X), where 'group_col'
        # indicates which distribution data came from
        if isinstance(self.cd_estimator, Oracle):
            # test graphically if Y is d-separated from F-node given Z
            test_stat, pvalue = self.cd_estimator.test(
                data, {group_col}, Y, Z, **self.cd_estimator_kwargs
            )
        else:
            # test statistically (Y || F-node | Z), or P(Y|Z) =? P'(Y|Z)
            test_stat, pvalue = self.cd_estimator.test(
                data, Z, Y, group_col, **self.cd_estimator_kwargs
            )

        self.n_ci_tests += 1
        return test_stat, pvalue

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        """Override the computation for conditioning sets.

        Parameters
        ----------
        adj_graph : nx.Graph
            _description_
        x_var : Column
            _description_
        y_var : Column
            _description_

        Returns
        -------
        Z : Set[Column]
            _description_
        """
        f_nodes = self.context_.f_nodes

        # if F-nodes is not defined, then we are simply doing learning in the observational setting
        if len(f_nodes) == 0:
            # compute the possible variables used in the conditioning set
            return super()._compute_candidate_conditioning_sets(adj_graph, x_var, y_var)
        else:
            if y_var in f_nodes:
                raise RuntimeError("This should not be the case")

            # get only neighboring sets of Y-vars, or PDS that depend on Y
            possible_variables = set(adj_graph.neighbors(y_var))
            return possible_variables

    def _learn_skeleton_with_interventions(self, interv_data: List[pd.DataFrame], context: Context):
        self.context_ = make_context(context, create_using=InterventionalContextBuilder).build()

        # get the initialized graph
        adj_graph = self.context_.init_graph
        f_nodes = self.context_.f_nodes

        print("Starting learning int skeleton: ", f_nodes)

        # the size of the conditioning set will start off at the minimum
        size_cond_set = self.min_cond_set_size_

        # track progress of the algorithm for which edges to remove to ensure stability
        self.remove_edges = set()

        # Outer loop: iterate over 'size_cond_set' until stopping criterion is met
        # - 'size_cond_set' > 'max_cond_set_size' or
        # - All (X, Y) pairs have candidate conditioning sets of size < 'size_cond_set'
        while 1:
            cont = False
            # initialize set of edges to remove at the end of every loop
            self.remove_edges = set()

            # loop through every node
            for x_var in f_nodes:
                possible_adjacencies = set(adj_graph.neighbors(x_var))

                logger.info(f"Considering node {x_var}...\n\n")

                for y_var in possible_adjacencies:
                    # a node cannot be a parent to itself in DAGs
                    if y_var == x_var:
                        continue

                    # if Y is also a F-node, then they are automatically assumed d-separated
                    if y_var in f_nodes:
                        pvalue = 1.0
                        test_stat = 0.0

                        # post-process the CI test results
                        removed_edge = self._postprocess_ci_test(
                            adj_graph, x_var, y_var, set(), test_stat, pvalue
                        )

                        # summarize the comparison of XY
                        self._summarize_xy_comparison(x_var, y_var, removed_edge, pvalue)
                        continue

                    # TODO: allow ignoring fixed edges
                    # if self.context_.included_edges.has_edge(x_var, y_var):
                    #     continue
                    # if self.context_.excluded_edges.has_edge(x_var, y_var):
                    #     pvalue = 1.0
                    #     test_stat = 0.0

                    #     # post-process the CI test results
                    #     removed_edge = self._postprocess_ci_test(
                    #         adj_graph, x_var, y_var, {}, test_stat, pvalue
                    #     )

                    #     # summarize the comparison of XY
                    #     self._summarize_xy_comparison(x_var, y_var, removed_edge, pvalue)
                    #     continue

                    # compute the possible variables used in the conditioning set
                    possible_variables = self._compute_candidate_conditioning_sets(
                        adj_graph,
                        x_var,
                        y_var,
                    )

                    logger.debug(
                        f"Adj({x_var}) without {y_var} with size={len(possible_adjacencies) - 1} "
                        f"with p={size_cond_set}. The possible variables to condition on are: "
                        f"{possible_variables}."
                    )

                    # check that number of adjacencies is greater then the
                    # cardinality of the conditioning set
                    if len(possible_variables) < size_cond_set:
                        logger.debug(
                            f"\n\nBreaking for {x_var}, {y_var}, {len(possible_adjacencies)}, "
                            f"{size_cond_set}, {possible_variables}"
                        )
                        continue
                    else:
                        cont = True

                    # generate iterator through the conditioning sets
                    conditioning_sets = _iter_conditioning_set(
                        possible_variables=possible_variables,
                        x_var=x_var,
                        y_var=y_var,
                        size_cond_set=size_cond_set,
                    )

                    # now iterate through the possible parents
                    for comb_idx, cond_set in enumerate(conditioning_sets):
                        # check the number of combinations of possible parents we have tried
                        # to use as a separating set
                        if (
                            self.max_combinations_ is not None
                            and comb_idx >= self.max_combinations_
                        ):
                            break

                        # compute conditional independence test
                        test_stat, pvalue = self.evaluate_fnode_edge(
                            interv_data, {x_var}, {y_var}, set(cond_set)
                        )

                        # if any "independence" is found through inability to reject
                        # the null hypothesis, then we will break the loop comparing X and Y
                        # and say X and Y are conditionally independent given 'cond_set'
                        if pvalue > self.alpha:
                            break

                    # post-process the CI test results
                    removed_edge = self._postprocess_ci_test(
                        adj_graph, x_var, y_var, cond_set, test_stat, pvalue
                    )

                    # summarize the comparison of XY
                    self._summarize_xy_comparison(x_var, y_var, removed_edge, pvalue)

            # finally remove edges after performing
            # conditional independence tests
            logger.info(f"For p = {size_cond_set}, removing all edges: {self.remove_edges}")

            # Remove non-significant links
            # Note: Removing edges at the end ensures "stability" of the algorithm
            # with respect to the randomness choice of pairs of edges considered in the inner loop
            adj_graph.remove_edges_from(self.remove_edges)

            # increment the conditioning set size
            size_cond_set += 1

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > self.max_cond_set_size_ or cont is False:
                break

        self.adj_graph_ = adj_graph

    def _learn_skeleton_with_observations(self, obs_data: pd.DataFrame, context: Context):
        f_nodes = context.f_nodes

        # get the init graph that does not contain any F-nodes
        obs_context_bld = make_context(context, create_using=ContextBuilder)
        init_graph = deepcopy(context.init_graph)

        # get the subgraph of non-f nodes
        non_f_nodes = set(init_graph.nodes) - set(f_nodes)
        obs_init_graph = init_graph.subgraph(non_f_nodes)

        # now learn the observational subgraph
        obs_context_bld.init_graph(obs_init_graph)
        obs_context = obs_context_bld.build()
        super().fit(obs_data, obs_context)

    def fit(self, data: List[pd.DataFrame], context: Context) -> None:
        """Fit data and context.

        Parameters
        ----------
        data : List[pd.DataFrame]
            List of dataframes corresponding to different distributions of data.
        context : Context
            Context object.
        """
        if self.cd_estimator_kwargs is None:
            self.cd_estimator_kwargs = dict()

        # error-check the datasets passed in match the intervention contexts
        if len(data) != context.num_distributions:
            raise RuntimeError(
                f"The number of datasets does not match the number of interventions. "
                f"You passed in {len(data)} different datasets, whereas "
                f"there are {len(context.intervention_targets)} different interventions "
                f"specified. It is assumed that the first dataset is observational, "
                f"while the rest are interventional."
            )

        orig_context = context.copy()
        f_nodes = context.f_nodes

        # it is fine to run the first stage of the FCI algorithm, as this will
        # not result in removing any edges among the F-nodes
        obs_data = data[0]
        self._learn_skeleton_with_observations(obs_data, context)

        # keep track of the observational skeleton graph
        obs_skel_graph = self.adj_graph_.copy()

        # all separating sets are either:
        # i) augmented with all F-nodes, or
        # ii) augmented with all F-nodes except intervention index 'i'
        # R9 allows us to leverage F-nodes being not in separating sets to

        # augment all separating sets that have non-empty sets with all
        # F-nodes to keep consistency with the algorithm
        for x_var, y_vars in self.sep_set_.items():
            for y_var in y_vars:
                sep_sets: List = self.sep_set_.get(x_var).get(y_var)  # type: ignore
                if len(sep_sets) > 0:
                    for idx in range(len(sep_sets)):
                        self.sep_set_[x_var][y_var][idx].update(f_nodes)

        # index all datasets, where the first one may be observational
        non_f_nodes = self.context_.get_non_f_nodes()

        # reset the init graph and this time learn the skeleton using
        # interventional distributions
        # create a complete subgraph of F-nodes with all other nodes
        for node in f_nodes:
            for obs_node in set(f_nodes).union(set(non_f_nodes)):
                if node == obs_node:
                    continue
                self.adj_graph_.add_edge(node, obs_node, test_stat=np.inf, pvalue=-1e-5)

        # reset context and add observational skeleton
        self.context_ = (
            make_context(orig_context, create_using=InterventionalContextBuilder)
            .init_graph(self.adj_graph_.copy())
            .build()
        )
        self.context_.add_state_variable("obs_skel_graph", obs_skel_graph)

        # now, we'll fit the data using interventional data by looping over all
        # combinations of F-nodes and their neighbors
        self._learn_skeleton_with_interventions(data, self.context_)
