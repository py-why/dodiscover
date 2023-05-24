from typing import FrozenSet, List, Optional, Tuple

import networkx as nx
import pandas as pd

from dodiscover._protocol import EquivalenceClass
from dodiscover.cd import BaseConditionalDiscrepancyTest
from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.constraint.config import ConditioningSetSelection
from dodiscover.typing import Column, SeparatingSet

from ..context import Context
from .intervention import PsiFCI
from .skeleton import LearnMultiDomainSkeleton


class SFCI(PsiFCI):
    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        cd_estimator: BaseConditionalDiscrepancyTest,
        alpha: float = 0.05,
        min_cond_set_size: Optional[int] = None,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        condsel_method: ConditioningSetSelection = ConditioningSetSelection.NBRS,
        apply_orientations: bool = True,
        keep_sorted: bool = False,
        max_iter: int = 1000,
        max_path_length: Optional[int] = None,
        pds_condsel_method: ConditioningSetSelection = ConditioningSetSelection.PDS,
        n_jobs: Optional[int] = None,
        debug: bool = False,
    ):
        super().__init__(
            ci_estimator,
            cd_estimator,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            condsel_method,
            apply_orientations,
            keep_sorted,
            max_iter,
            max_path_length,
            pds_condsel_method,
            n_jobs=n_jobs,
        )
        self.debug = debug

    def learn_skeleton(
        self, data: pd.DataFrame, context: Context, sep_set: Optional[SeparatingSet] = None
    ) -> Tuple[nx.Graph, SeparatingSet]:
        # now compute all possibly d-separating sets and learn a better skeleton
        self.skeleton_learner_ = LearnMultiDomainSkeleton(
            self.ci_estimator,
            self.cd_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            condsel_method=self.condsel_method,
            second_stage_condsel_method=self.pds_condsel_method,
            keep_sorted=False,
            max_path_length=self.max_path_length,
            n_jobs=self.n_jobs,
        )
        self.skeleton_learner_.fit(
            data, context, self.domain_indices, self.intervention_targets, debug=self.debug
        )

        self.context_ = self.skeleton_learner_.context_.copy()
        skel_graph = self.skeleton_learner_.adj_graph_
        sep_set = self.skeleton_learner_.sep_set_
        self.n_ci_tests += self.skeleton_learner_.n_ci_tests
        return skel_graph, sep_set

    def fit(self, data: List[pd.DataFrame], context: Context, domain_indices, intervention_targets):
        """Learn the relevant causal graph equivalence class.

        From the pairs of datasets, we take all combinations and
        construct F-nodes corresponding to those.

        Parameters
        ----------
        data : List[pd.DataFrame]
            The list of different datasets assigned to different
            environments. We assume the first dataset is always
            observational.
        context : Context
            The context with interventional assumptions.

        Returns
        -------
        self : PsiFCI
            The fitted learner.
        """
        if not isinstance(data, list):
            raise RuntimeError("The input datasets must be in a Python list.")

        # n_datasets = len(data)
        # n_distributions = context.num_distributions

        # if n_datasets != n_distributions:
        #     raise RuntimeError(
        #         f"There are {n_datasets} passed in, but {n_distributions} "
        #         f"total assumed distributions. There must be a matching number of datasets and "
        #         f"'context.num_distributions'."
        #     )
        self.domain_indices = domain_indices
        self.intervention_targets = intervention_targets

        return super().fit(data, context)

    def _apply_rule11(self, graph: EquivalenceClass, context: Context) -> Tuple[bool, List]:
        augmented_nodes = context.f_nodes + context.s_nodes

        oriented_edges = []
        added_arrows = True
        for node in augmented_nodes:
            for nbr in graph.neighbors(node):
                if nbr in augmented_nodes:
                    continue

                # remove all edges between node and nbr and orient this out
                graph.remove_edge(node, nbr)
                graph.remove_edge(nbr, node)
                graph.add_edge(node, nbr, graph.directed_edge_name)
                oriented_edges.append((node, nbr))
        return added_arrows, oriented_edges

    def _apply_rule12(
        self,
        graph: EquivalenceClass,
        u: Column,
        a: Column,
        c: Column,
        context: Context,
    ) -> bool:
        """Apply "Rule 9" of the I-FCI algorithm.

        Checks for inducing paths where 'u' is the F-node, and 'a' and 'c' are connected:

        'u' -> 'a' *-* 'c' with 'u' -> 'c', then orient 'a' -> 'c'.

        For original details of the rule, see :footcite:`Kocaoglu2019characterization`.

        Parameters
        ----------
        graph : EquivalenceClass
            The causal graph.
        u : Column
            The candidate F-node
        a : Column
            Neighbors of the F-node.
        c : Column
            Neighbors of the F-node.
        symmetric_diff_map : dict
            A mapping from the F-nodes to the symmetric difference of the pair of
            intervention targets each F-node represents. I.e. if F-node, F1 represents
            the pair of intervention distributions with targets {'x'}, and {'x', 'y'},
            then F1 maps to {'y'} in the symmetric diff map.

        Returns
        -------
        added_arrows : bool
            Whether or not an orientation was made.

        References
        ----------
        .. footbibliography::
        """
        f_nodes = context.f_nodes
        symmetric_diff_map = context.symmetric_diff_map

        added_arrows = False
        if u in f_nodes and self.known_intervention_targets:
            # get sigma map to map F-node to its symmetric difference target
            S_set: FrozenSet = symmetric_diff_map.get(u, frozenset())

            # check domain
            domains_u = context.domain_map[u]

            # check the presence of an S-node for that domain
            if len(domains_u) == 2:
                for s_node in context.s_nodes:
                    if context.domain_map[s_node] == domains_u:
                        # check if the s-node is d-connected to F
                        if graph.has_edge(s_node, c):
                            return False

            # now, we know that there is no S-node for the domain of u
            # that will alter the distribution of a/c, so we check for
            # an inducing path that we can orient properly
            # check a *-* c
            if (
                len(S_set) == 1
                and a in S_set
                and (graph.has_edge(a, c) or graph.has_edge(c, a))
                and graph.has_edge(u, a)
                and graph.has_edge(u, c)
            ):
                # remove all edges between a and c
                graph.remove_edge(a, c)
                graph.remove_edge(c, a)

                # then orient X -> Y
                graph.add_edge(a, c, graph.directed_edge_name)

                added_arrows = True
        return added_arrows

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClass:
        import pywhy_graphs as pgraph

        # convert the undirected skeleton graph to its PAG-class, where
        # all left-over edges have a "circle" endpoint
        pag = pgraph.AugmentedPAG(incoming_circle_edges=graph, name="SPAG derived with S-FCI")

        # get the graph attributes
        pag.graph = graph.graph

        # XXX: assign targets as well
        # assign f-nodes
        # for f_node in self.context_.f_nodes:
        #     pag.set_f_node(f_node)
        # for s_node in self.context_.s_nodes:
        #     domain_ids = self.context_.domain_map[s_node]
        #     pag.add_s_node(s_node, domain_ids=domain_ids, node_changes=)
        return pag
