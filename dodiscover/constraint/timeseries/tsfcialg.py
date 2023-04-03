from typing import Optional, Tuple

import networkx as nx
import pandas as pd

from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.constraint.timeseries import (
    LearnTimeSeriesSemiMarkovianSkeleton,
    LearnTimeSeriesSkeleton,
)
from dodiscover.context import Context
from dodiscover.context_builder import make_ts_context
from dodiscover.typing import SeparatingSet

from ..._protocol import EquivalenceClass
from ..config import SkeletonMethods
from ..fcialg import FCI


class TimeSeriesFCI(FCI):
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
        max_path_length: Optional[int] = None,
        selection_bias: bool = False,
        pds_skeleton_method: SkeletonMethods = SkeletonMethods.PDS,
        separate_lag_phase: bool = False,
        contemporaneous_edges: bool = True,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            apply_orientations,
            max_iter,
            max_path_length,
            selection_bias,
            pds_skeleton_method,
            **ci_estimator_kwargs,
        )
        self.separate_lag_phase = separate_lag_phase
        self.contemporaneous_edges = contemporaneous_edges

    def learn_skeleton(
        self, data: pd.DataFrame, context: Context, sep_set: Optional[SeparatingSet] = None
    ) -> Tuple[nx.Graph, SeparatingSet]:
        import pywhy_graphs

        from dodiscover import make_ts_context

        # initially learn the skeleton
        skel_alg = LearnTimeSeriesSkeleton(
            self.ci_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.skeleton_method,
            keep_sorted=False,
            separate_lag_phase=self.separate_lag_phase,
            contemporaneous_edges=self.contemporaneous_edges,
            **self.ci_estimator_kwargs,
        )
        skel_alg.fit(data, context)

        skel_graph = skel_alg.adj_graph_
        sep_set = skel_alg.sep_set_
        self.n_ci_tests += skel_alg.n_ci_tests

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = pywhy_graphs.StationaryTimeSeriesPAG(
            incoming_circle_edges=skel_graph, name="PAG derived with tsFCI"
        )

        # orient colliders
        self.orient_unshielded_triples(pag, sep_set)

        # convert the adjacency graph
        new_init_graph = pag.to_ts_undirected()

        # Update the Context:
        # add the corresponding intermediate PAG now to the context
        # new initialization graph
        for (_, _, d) in new_init_graph.edges(data=True):
            if "test_stat" in d:
                d.pop("test_stat")
            if "pvalue" in d:
                d.pop("pvalue")
        context = (
            make_ts_context(context).init_graph(new_init_graph).state_variable("PAG", pag).build()
        )

        # # now compute all possibly d-separating sets and learn a better skeleton
        skel_alg = LearnTimeSeriesSemiMarkovianSkeleton(
            self.ci_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.pds_skeleton_method,
            keep_sorted=False,
            max_path_length=self.max_path_length,
            separate_lag_phase=self.separate_lag_phase,
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
        pass

    def orient_contemporaneous_edges(self, graph: EquivalenceClass):
        pass

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClass:
        return super().convert_skeleton_graph(graph)
