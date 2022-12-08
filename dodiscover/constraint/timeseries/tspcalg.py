import logging
from itertools import combinations, permutations
from typing import Iterator, Optional, Tuple

import networkx as nx
import pandas as pd

from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.context import Context
from dodiscover.context_builder import make_ts_context
from dodiscover.typing import SeparatingSet

from ..._protocol import EquivalenceClass
from ..config import SkeletonMethods
from ..pcalg import PC
from .skeleton import LearnTimeSeriesSkeleton

logger = logging.getLogger()


class TimeSeriesPC(PC):
    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        min_cond_set_size: Optional[int] = None,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        apply_orientations: bool = True,
        max_iter: int = 1000,
        separate_lag_phase: bool = False,
        contemporaneous_edges: bool = True,
        **ci_estimator_kwargs,
    ):
        """[Experimental] Time-series PC algorithm.

        A PC algorithm specialized for time-series, which differs in two ways:
        1. learning the skeleton: during the removal of a non-contemporaneous edge,
        remove all corresponding homologous edges.
        2. orienting edges: during the orientation of a non-contemporaneous edge,
        remove all corresponding homologous edges.

        Homologous edges are edges that have repeating structure over time.

        Parameters
        ----------
        ci_estimator : BaseConditionalIndependenceTest
            _description_
        alpha : float, optional
            _description_, by default 0.05
        min_cond_set_size : Optional[int], optional
            _description_, by default None
        max_cond_set_size : Optional[int], optional
            _description_, by default None
        max_combinations : Optional[int], optional
            _description_, by default None
        skeleton_method : SkeletonMethods, optional
            _description_, by default SkeletonMethods.NBRS
        apply_orientations : bool, optional
            _description_, by default True
        max_iter : int, optional
            _description_, by default 1000
        contemporaneous_edges : bool
            Whether or not to assume contemporaneous edges.

        References
        ----------
        .. footbibliography::
        """
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            apply_orientations,
            max_iter,
            **ci_estimator_kwargs,
        )
        self.separate_lag_phase = separate_lag_phase
        self.contemporaneous_edges = contemporaneous_edges

    def learn_skeleton(
        self,
        data: pd.DataFrame,
        context: Context,
        sep_set: Optional[SeparatingSet] = None,
    ) -> Tuple[nx.Graph, SeparatingSet]:
        skel_alg = LearnTimeSeriesSkeleton(
            self.ci_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.skeleton_method,
            keep_sorted=False,
            separate_lag_phase=False,
            contemporaneous_edges=self.contemporaneous_edges,
            **self.ci_estimator_kwargs,
        )
        skel_alg.fit(data, context)

        skel_graph = skel_alg.adj_graph_
        sep_set = skel_alg.sep_set_
        self.n_ci_tests += skel_alg.n_ci_tests
        return skel_graph, sep_set

    def fit(self, data: pd.DataFrame, context: Context) -> None:
        self.context_ = make_ts_context(context).build()
        graph = self.context_.init_graph
        self.init_graph_ = graph
        self.fixed_edges_ = self.context_.included_edges

        # create a reference to the underlying data to be used
        self.X_ = data

        # initialize graph object to apply learning
        self.separating_sets_ = self._initialize_sep_sets(self.init_graph_)

        # learn skeleton graph and the separating sets per variable
        graph, self.separating_sets_ = self.learn_skeleton(
            self.X_, self.context_, self.separating_sets_
        )

        # convert networkx.Graph to relevant causal graph object
        graph = self.convert_skeleton_graph(graph)

        # orient edges on the causal graph object
        if self.apply_orientations:
            # first orient lagged edges
            self.orient_lagged_edges(graph)

            if self.contemporaneous_edges:
                # next orient contemporaneous edges if necessary
                self.orient_contemporaneous_edges(graph)

        # store resulting data structures
        self.graph_ = graph

    def orient_lagged_edges(self, graph: EquivalenceClass):
        undirected_subgraph = graph.get_graphs(graph.undirected_edge_name)
        # get non-lag nodes
        for node in undirected_subgraph.nodes_at(t=0):
            # get all lagged nbrs
            for nbr in undirected_subgraph.lagged_neighbors(node):
                # now orient this edge as u -> v
                graph.orient_uncertain_edge(nbr, node)

    def orient_contemporaneous_edges(self, graph):
        # for all pairs of non-adjacent variables with a common neighbor
        # check if we can orient the edge as a collider
        self.orient_unshielded_triples(graph, self.separating_sets_)
        self.orient_edges(graph)

    def _orientable_edges(self, graph: EquivalenceClass) -> Iterator:
        for (i, j) in permutations(graph.nodes_at(t=0), 2):  # type: ignore
            if i == j:
                continue
            yield (i, j)

    def _orientable_triples(self, graph: EquivalenceClass) -> Iterator:
        # for every node in the PAG, evaluate neighbors that have any edge
        for u in graph.nodes_at(t=0):  # type: ignore
            for v_i, v_j in combinations(graph.neighbors(u), 2):
                yield (v_i, u, v_j)

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClass:
        from pywhy_graphs import StationaryTimeSeriesCPDAG

        # convert Graph object to a CPDAG object with
        # all undirected edges
        graph = StationaryTimeSeriesCPDAG(incoming_undirected_edges=graph)
        return graph
