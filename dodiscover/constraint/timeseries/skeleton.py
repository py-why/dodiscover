import logging
from itertools import chain
from typing import Iterator, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover._protocol import TimeSeriesGraph
from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.context import Context, TimeSeriesContext
from dodiscover.typing import Column, SeparatingSet

from ...context_builder import make_ts_context
from ..config import SkeletonMethods
from ..skeleton import LearnSkeleton
from ..utils import _iter_conditioning_set
from .utils import convert_ts_df_to_multiindex

logger = logging.getLogger()


def nodes_in_time_order(G: TimeSeriesGraph) -> Iterator:
    """Return nodes from G in time order starting from max-lag to t=0."""
    for t in range(G.max_lag, -1, -1):
        for node in G.nodes_at(t):
            yield node


class LearnTimeSeriesSkeleton(LearnSkeleton):
    """Learn a skeleton time-series graph from a Markovian causal model.

    Learning time-series causal graph skeletons is a more complex task compared to its
    iid counterpart. There are a few key differences to be aware of:

    1. Without extending the maximum-lag, there is always latent confounding: Consider as
    an example, two variable lag-1 ts-causal-graph.

          t-1   t
        X  o -> o
        Y  o -> o

    with Y(t-1) -> X(t). Assuming stationarity, then

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
    separate_lag_phase : bool,
        Whether or not to separate the lagged and contemporaneous skeleton learning
        phase. If False (default), then will test all CI dependences in the same loop.
    contemporaneous_edges : bool,
        Whether or not there are contemporaneous edges (i.e. edges that occur at the same time point
        between two nodes). By default is True.
    """

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
        max_path_length: Optional[int] = None,
        separate_lag_phase: bool = False,
        contemporaneous_edges: bool = True,
        **ci_estimator_kwargs,
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
            **ci_estimator_kwargs,
        )
        self.max_path_length = max_path_length
        self.separate_lag_phase = separate_lag_phase
        self.contemporaneous_edges = contemporaneous_edges

    def evaluate_edge(
        self, data: pd.DataFrame, X: Column, Y: Column, Z: Optional[Set[Column]] = None
    ) -> Tuple[float, float]:
        """Evaluate an edge, but the data frame has columns as '<node_name>_<lag>'."""
        return super().evaluate_edge(data, X, Y, Z)

    def _learn_skeleton(
        self, data: pd.DataFrame, adj_graph: TimeSeriesGraph, nbr_search: str = "all"
    ):
        """Private method to learn a skeleton"""
        # the size of the conditioning set will start off at the minimum
        size_cond_set = self.min_cond_set_size_

        # If there is latent confounding, we need to test all nodes starting from
        # max-lag. Because adjacencies are only repeated backwards in time
        testable_nodes = list(nodes_in_time_order(adj_graph))

        # to do causal discovery of time-series graphs,
        # homologous edges should not be removed automatically
        adj_graph.set_auto_removal("forwards")

        # # to do causal discovery of time-series graphs,
        # # homologous edges should not be removed automatically
        # adj_graph.set_auto_removal("backwards")

        print(f"Testing nodes in the following order for learning skeleton: {testable_nodes}")

        # Outer loop: iterate over 'size_cond_set' until stopping criterion is met
        # - 'size_cond_set' > 'max_cond_set_size' or
        # - All (X, Y) pairs have candidate conditioning sets of size < 'size_cond_set'
        while 1:
            cont = False
            # initialize set of edges to remove at the end of every loop
            self.remove_edges = set()

            # loop through every node
            # Note: in time-series graphs, all nodes are defined as a 2-tuple
            # of (<node>, <lag>)
            for y_var in testable_nodes:
                # we only consider variables with required lag
                # if y_var[1] != 0:
                #     continue

                # TODO: need more efficient way of querying all possible edges
                if nbr_search == "all":
                    lagged_nbrs = adj_graph.lagged_neighbors(y_var)
                    contemporaneous_nbrs = adj_graph.contemporaneous_neighbors(y_var)
                    possible_adjacencies = list(set(lagged_nbrs).union(set(contemporaneous_nbrs)))
                elif nbr_search == "lagged":
                    possible_adjacencies = adj_graph.lagged_neighbors(y_var)
                elif nbr_search == "contemporaneous":
                    possible_adjacencies = adj_graph.contemporaneous_neighbors(y_var)

                logger.info(f"Considering node {y_var}...\n\n")
                print(f"\n\nTesting {y_var} against possible adjacencies {possible_adjacencies}")
                print(f"size conditioning set p = {size_cond_set}")
                for x_var in possible_adjacencies:
                    # a node cannot be a parent to itself in DAGs
                    if y_var == x_var:
                        continue

                    # ignore fixed edges
                    if (x_var, y_var) in self.context.included_edges.edges:
                        continue

                    # compute the possible variables used in the conditioning set
                    possible_variables = self._compute_candidate_conditioning_sets(
                        adj_graph,
                        y_var,
                        x_var,
                        skeleton_method=self.skeleton_method,
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
                            print(f"Removing {x_var} - {y_var} with {cond_set}.")
                            break

                    # post-process the CI test results
                    removed_edge = self._postprocess_ci_test(
                        adj_graph, x_var, y_var, cond_set, test_stat, pvalue
                    )

                    # summarize the comparison of XY
                    self._summarize_xy_comparison(x_var, y_var, removed_edge, pvalue)

            # finally remove edges after performing
            # conditional independence tests
            logger.info("\n---------------------------------------------")
            logger.info(f"For p = {size_cond_set}, removing all edges: {self.remove_edges}")

            # TODO: should not hack the removal of edges to remove
            from_set = []
            to_set = []
            for u, v in self.remove_edges:
                # the opposite is already in there...
                if v in to_set and u in from_set:
                    continue
                from_set.append(u)
                to_set.append(v)
            self.remove_edges = set()
            for u, v in zip(from_set, to_set):
                self.remove_edges.add((u, v))

            # Remove non-significant links
            # Note: Removing edges at the end ensures "stability" of the algorithm
            # with respect to the randomness choice of pairs of edges considered in the inner loop
            print(f"Removing edges {self.remove_edges}")
            adj_graph.remove_edges_from(list(self.remove_edges))

            # increment the conditioning set size
            size_cond_set += 1

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > self.max_cond_set_size_ or cont is False:
                break

        return adj_graph

    def fit(self, data: pd.DataFrame, context: Context) -> None:
        """Run structure learning to learn the skeleton of the causal graph.

        Parameters
        ----------
        data : pd.DataFrame
            The data to learn the causal graph from.
        context : Context
            A context object.
        """
        if self.separate_lag_phase and not self.contemporaneous_edges:
            raise ValueError(
                "There is assumed no contemporaneous edges, but you also "
                "specified to separate the lag and contemporaneous phase."
            )

        data, context = self._preprocess_data(data, context)
        self.context = (
            make_ts_context(context)
            .observed_variables(data.columns.get_level_values("variable").tolist())
            .build()
        )

        # initialize learning parameters
        self._initialize_params()

        # get the initialized graph
        adj_graph: TimeSeriesGraph = self.context.init_graph

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

        # learn the skeleton graph
        if self.separate_lag_phase:
            # first do the lagged search
            adj_graph = self._learn_skeleton(data, adj_graph, nbr_search="lagged")

            # then do contemporaneous
            adj_graph = self._learn_skeleton(data, adj_graph, nbr_search="contemporaneous")
        else:
            adj_graph = self._learn_skeleton(data, adj_graph, nbr_search="all")

        # possibly remove all contemporaneous edges if there is
        # no assumption of contemporaneous causal structure
        # TODO: can make sure we don't inner-test the CI relations between contemporaneous edges
        if not self.contemporaneous_edges:
            auto_removal = adj_graph._auto_removal  # type: ignore
            adj_graph.set_auto_removal(None)
            for u, v in adj_graph.edges:  # type: ignore
                if u[1] == v[1]:
                    adj_graph.remove_edge(u, v)  # type: ignore
            adj_graph.set_auto_removal(auto_removal)

        self.adj_graph_ = adj_graph

    def _preprocess_data(self, data: pd.DataFrame, context: TimeSeriesContext):
        """Preprocess data and context.

        In time-series causal discovery, dataframe of the shape (n_times, n_signals)
        are re-formatted to a dataframe with (n_samples, n_signals x lag_points)
        with a multi-index column with variable names at level 1 and lag time
        points at level 2. For example, a multi-index column of ``[('x', 0), ('x', -1)]``
        would indicate the first column is the variable x at lag 0 and the second
        column is variable x at lag 1.
        """
        # first reformat data
        max_lag = context.max_lag  # type: ignore
        data = convert_ts_df_to_multiindex(data, max_lag)

        # run preprocessing
        if set(context.observed_variables) != set(data.columns.get_level_values("variable")):
            raise RuntimeError(
                "The observed variable names in data and context do not match: \n"
                f"- {context.observed_variables} \n"
                f"- {data.columns}"
            )
        edge_attrs = set(
            chain.from_iterable(d.keys() for *_, d in context.init_graph.edges(data=True))
        )
        if "test_stat" in edge_attrs or "pvalue" in edge_attrs:
            raise RuntimeError(
                "Running skeleton discovery with adjacency graph "
                "with 'test_stat' or 'pvalue' is not supported yet."
            )

        return data, context


class LearnTimeSeriesSemiMarkovianSkeleton(LearnTimeSeriesSkeleton):
    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.PDS_T,
        keep_sorted: bool = False,
        max_path_length: Optional[int] = None,
        separate_lag_phase: bool = False,
        contemporaneous_edges: bool = True,
        **ci_estimator_kwargs,
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
            **ci_estimator_kwargs,
        )
        if max_path_length is None:
            max_path_length = np.inf
        self.max_path_length = max_path_length
        self.separate_lag_phase = separate_lag_phase
        self.contemporaneous_edges = contemporaneous_edges

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column, skeleton_method: SkeletonMethods
    ) -> Set[Column]:
        import pywhy_graphs as pgraph

        # get PAG from the context object
        pag = self.context.state_variable("PAG")

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
        elif skeleton_method == SkeletonMethods.PDS_T:
            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            possible_variables = pgraph.pds_t(
                pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
            )
        elif skeleton_method == SkeletonMethods.PDS_T_PATH:
            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            possible_variables = pgraph.pds_t_path(
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

    def fit(self, data: pd.DataFrame, context: TimeSeriesContext) -> None:
        return super().fit(data, context)
