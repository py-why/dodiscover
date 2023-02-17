import logging
from itertools import permutations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

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
    ci_estimator : Callable
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
    skeleton_method : SkeletonMethods
        The method to use for testing conditional independence. Must be one of
        ('neighbors', 'complete', 'neighbors_path'). See Notes for more details.
    apply_orientations : bool
        Whether or not to apply orientation rules given the learned skeleton graph
        and separating set per pair of variables. If ``True`` (default), will
        apply Zhang's orientation rules R0-10, orienting colliders and certain
        arrowheads and tails :footcite:`Zhang2008`.
    max_iter : int
        The maximum number of iterations through the graph to apply
        orientation rules.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    pds_skeleton_method : SkeletonMethods
        The method to use for learning the skeleton using PDS. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    known_intervention_targets : bool, optional
        If `True`, then will run the I-FCI algorithm. If `False`, will run the
        Psi-FCI algorithm. By default False.

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
        skeleton_method: ConditioningSetSelection = ConditioningSetSelection.NBRS,
        apply_orientations: bool = True,
        max_iter: int = 1000,
        max_path_length: Optional[int] = None,
        pds_skeleton_method: ConditioningSetSelection = ConditioningSetSelection.PDS,
        known_intervention_targets: bool = False,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            apply_orientations,
            max_iter=max_iter,
            max_path_length=max_path_length,
            selection_bias=False,
            pds_skeleton_method=pds_skeleton_method,
        )
        self.cd_estimator = cd_estimator
        self.known_intervention_targets = known_intervention_targets

    def learn_skeleton(
        self, data: pd.DataFrame, context: Context, sep_set: Optional[SeparatingSet] = None
    ) -> Tuple[nx.Graph, SeparatingSet]:
        # now compute all possibly d-separating sets and learn a better skeleton
        self.skeleton_learner_ = LearnInterventionSkeleton(
            self.ci_estimator,
            self.cd_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.skeleton_method,
            second_stage_skeleton_method=self.pds_skeleton_method,
            keep_sorted=False,
            max_path_length=self.max_path_length,
        )
        print(context)
        self.skeleton_learner_.fit(data, context)

        skel_graph = self.skeleton_learner_.adj_graph_
        sep_set = self.skeleton_learner_.sep_set_
        self.n_ci_tests += self.skeleton_learner_.n_ci_tests
        return skel_graph, sep_set

    def fit(self, data: List[pd.DataFrame], context: Context):
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
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if not isinstance(data, list):
            raise RuntimeError("The input datasets must be in a Python list.")

        n_datasets = len(data)
        n_distributions = context.num_distributions

        if n_datasets != n_distributions:
            raise RuntimeError(
                f"There are {n_datasets} passed in, but {n_distributions} "
                f"total assumed distributions. There must be a matching number of datasets and "
                f"'context.num_distributions'."
            )

        super().fit(data, context)

    def _apply_rule11(self, graph: EquivalenceClass, f_nodes: List) -> Tuple[bool, List]:
        """Apply "Rule 8" in I-FCI algorithm, which we call Rule 11.

        This orients all edges out of F-nodes. So patterns of the form

        ``('F', 0) *-* 'x'`` will become ``('F', 0) -> 'x'``.

        For original details of the rule, see :footcite:`Kocaoglu2019characterization`.

        Parameters
        ----------
        graph : EquivalenceClass
            The causal graph to apply rules to.
        f_nodes : list
            The list of f-nodes within the graph.

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
        oriented_edges = []
        added_arrows = True
        for node in f_nodes:
            for nbr in graph.neighbors(node):
                if nbr in f_nodes:
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
        f_nodes: List,
        symmetric_diff_map: Dict[Any, FrozenSet],
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
        f_nodes = self.context_.f_nodes
        symmetric_diff_map = self.context_.symmetric_diff_map
        _ = self._apply_rule11(graph, f_nodes)

        while idx < self.max_iter and not finished:
            change_flag = False
            logger.info(f"Running R1-10 for iteration {idx}")

            for u in graph.nodes:
                for (a, c) in permutations(graph.neighbors(u), 2):
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
                    r12_add = self._apply_rule12(graph, u, a, c, f_nodes, symmetric_diff_map)

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
            pag = pgraph.IPAG(incoming_circle_edges=graph, name="IPAG derived with I-FCI")
        else:
            pag = pgraph.PsiPAG(incoming_circle_edges=graph, name="PsiPAG derived with Psi-FCI")
        return pag
