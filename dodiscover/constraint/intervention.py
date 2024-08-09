import logging
from itertools import permutations
from typing import FrozenSet, List, Optional, Tuple

import networkx as nx
import pandas as pd

from dodiscover._protocol import EquivalenceClass
from dodiscover.cd import BaseConditionalDiscrepancyTest
from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.context import Context
from dodiscover.typing import Column, SeparatingSet

from .config import ConditioningSetSelection
from .fcialg import FCI
from .skeleton import LearnInterventionSkeleton

logger = logging.getLogger()


class PsiFCI(FCI):
    """Interventional (Psi) FCI algorithm.

    The I-FCI (or Psi-FCI) algorithm is an algorithm that accepts
    multiple sets of data that may pertain to observational and/or
    multiple interventional datasets under a known (I-FCI), or unknown (Psi-FCI)
    intervention target setting. Our API consolidates them here under
    one class, but you can control the setting using our hyperparameter.
    See :footcite:`Kocaoglu2019characterization` for more information on
    I-FCI and :footcite:`Jaber2020causal` for more information on Psi-FCI.

    The Psi-FCI algorithm is complete for the Psi-PAG equivalence class.
    However, the I-FCI has not been shown to be complete for the I-PAG
    equivalence class. Note that the I-FCI algorithm may change without
    notice.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    cd_estimator : BaseConditionalDiscrepancyTest
        The conditional discrepancy test function.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int, optional
        Minimum size of the conditioning set, by default None, which will be set to '0'.
        Used to constrain the computation spent on the algorithm.
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
    condsel_method : ConditioningSetSelection
        The method to use for selecting the conditioning sets. Must be one of
        ('neighbors', 'complete', 'neighbors_path'). See Notes for more details.
    apply_orientations : bool
        Whether or not to apply orientation rules given the learned skeleton graph
        and separating set per pair of variables. If ``True`` (default), will
        apply Zhang's orientation rules R0-10, orienting colliders and certain
        arrowheads and tails :footcite:`Zhang2008`.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). The conditioning set is chosen lexographically
        based on the sorted test statistic values of 'ith Pa(X) -> X', for each possible
        parent node of 'X'. This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    max_iter : int
        The maximum number of iterations through the graph to apply
        orientation rules.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    pds_condsel_method : ConditioningSetSelection
        The method to use for selecting the conditioning sets using PDS. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    known_intervention_targets : bool, optional
        If `True`, then will run the I-FCI algorithm. If `False`, will run the
        Psi-FCI algorithm. By default False.
    n_jobs : int, optional
        The number of parallel jobs to run. If -1, then the number of jobs is set to
        the number of cores. If 1 is given, no parallel computing code is used at all,
        By default None, which means 1.

    Notes
    -----
    Selection bias is unsupported because it is still an active research area.
    """

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
        known_intervention_targets: bool = False,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            condsel_method,
            apply_orientations,
            keep_sorted=keep_sorted,
            max_iter=max_iter,
            max_path_length=max_path_length,
            selection_bias=False,
            pds_condsel_method=pds_condsel_method,
            n_jobs=n_jobs,
        )
        self.cd_estimator = cd_estimator
        self.known_intervention_targets = known_intervention_targets

    def learn_skeleton(
        self,
        data: pd.DataFrame,
        context: Optional[Context] = None,
        sep_set: Optional[SeparatingSet] = None,
        **params,
    ) -> Tuple[nx.Graph, SeparatingSet]:
        if context is None:
            # make a private Context object to store causal context used in this algorithm
            # store the context
            from dodiscover.context_builder import make_context

            context = make_context().build()

        # now compute all possibly d-separating sets and learn a better skeleton
        self.skeleton_learner_ = LearnInterventionSkeleton(
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
        self.skeleton_learner_.learn_graph(data, context)

        self.context_ = self.skeleton_learner_.context_.copy()
        skel_graph = self.skeleton_learner_.adj_graph_
        sep_set = self.skeleton_learner_.sep_set_
        self.n_ci_tests += self.skeleton_learner_.n_ci_tests
        return skel_graph, sep_set

    def learn_graph(self, data: List[pd.DataFrame], context: Optional[Context] = None):
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
        if context is None:
            # make a private Context object to store causal context used in this algorithm
            # store the context
            from dodiscover.context_builder import make_context

            context = make_context().build()

        if not isinstance(data, list):
            raise TypeError("The input datasets must be in a Python list.")

        n_datasets = len(data)
        n_distributions = context.num_distributions

        if n_datasets != n_distributions:
            raise RuntimeError(
                f"There are {n_datasets} passed in, but {n_distributions} "
                f"total assumed distributions. There must be a matching number of datasets and "
                f"'context.num_distributions'."
            )

        return super().learn_graph(data, context)

    def _apply_rule11(self, graph: EquivalenceClass, context: Context) -> Tuple[bool, List]:
        """Apply "Rule 8" in I-FCI algorithm, which we call Rule 11.

        This orients all edges out of F-nodes. So patterns of the form

        ``('F', 0) *-* 'x'`` will become ``('F', 0) -> 'x'``.

        For original details of the rule, see :footcite:`Kocaoglu2019characterization`.

        Parameters
        ----------
        graph : EquivalenceClass
            The causal graph to apply rules to.
        context : Context
            The causal context.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were added.
        oriented_edges : List
            A list of oriented edges.

        References
        ----------
        .. footbibliography::
        """
        augmented_nodes = context.get_augmented_nodes()

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
        self, graph: EquivalenceClass, u: Column, a: Column, c: Column, context: Context
    ) -> bool:
        """Apply orientation rule of the I-FCI algorithm.

        In the I-FCI algorithm, this is called "Rule 9". Checks for inducing paths where
        'u' is the F-node, and 'a' and 'c' are connected:

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
        context : Context
            The causal context.

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

    def _apply_orientation_rules(self, graph: EquivalenceClass, sep_set: SeparatingSet):
        idx = 0
        finished = False

        # apply R11, which is called R8 in I-FCI / Psi-FCI orienting all F-nodes
        _ = self._apply_rule11(graph, self.context_)

        while idx < self.max_iter and not finished:
            change_flag = False
            logger.info(f"Running R1-10 for iteration {idx}")

            for u in graph.nodes:
                for a, c in permutations(graph.neighbors(u), 2):
                    logger.debug(f"Check {u} {a} {c}")

                    # apply R1-3 to orient triples and arrowheads
                    r1_add = self._apply_rule1(graph, u, a, c)
                    r2_add = self._apply_rule2(graph, u, a, c)
                    r3_add = self._apply_rule3(graph, u, a, c)

                    # apply R4, orienting discriminating paths
                    r4_add, _ = self._apply_rule4(graph, u, a, c, sep_set)

                    # apply R8 to orient more tails
                    r8_add = self._apply_rule8(graph, u, a, c)

                    # apply R9-10 to orient uncovered potentially directed paths
                    r9_add, _ = self._apply_rule9(graph, a, u, c)

                    # a and c are neighbors of u, so u is the endpoint desired
                    r10_add, _, _ = self._apply_rule10(graph, a, c, u)

                    # apply R12, called R9 in I-FCI when we know the intervention targets
                    r12_add = self._apply_rule12(graph, u, a, c, self.context_)

                    # see if there was a change flag
                    all_flags = [r1_add, r2_add, r3_add, r4_add, r8_add, r9_add, r10_add, r12_add]
                    if any(all_flags) and not change_flag:
                        logger.info(f"{change_flag} with {all_flags}")
                        change_flag = True

            # check if we should continue or not
            if not change_flag:
                finished = True
                if not self.selection_bias:
                    logger.info(f"Finished applying R1-4, and R8-10 with {idx} iterations")
                if self.selection_bias:
                    logger.info(f"Finished applying R1-10 with {idx} iterations")
                break
            idx += 1

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClass:
        import pywhy_graphs as pgraph

        # convert the undirected skeleton graph to its PAG-class, where
        # all left-over edges have a "circle" endpoint
        if self.known_intervention_targets:
            pag = pgraph.AugmentedPAG(
                incoming_circle_edges=graph, name="AugmentedPAG derived with I-FCI"
            )
        else:
            pag = pgraph.AugmentedPAG(
                incoming_circle_edges=graph, name="AugmentedPAG derived with Psi-FCI"
            )

        # XXX: assign targets as well
        # assign f-nodes
        for f_node in self.context_.f_nodes:
            pag.set_f_node(f_node)
        return pag
