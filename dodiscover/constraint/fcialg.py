import logging
from itertools import combinations, permutations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import networkx as nx
import pandas as pd
from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.constraint.utils import is_in_sep_set
from dodiscover.typing import Column, SeparatingSet

from .._protocol import SemiMarkovianEquivalenceClass
from ._classes import BaseConstraintDiscovery

logger = logging.getLogger()


class FCI(BaseConstraintDiscovery):
    """The Fast Causal Inference (FCI) algorithm for causal discovery.

    A complete constraint-based causal discovery algorithm that
    operates on observational data :footcite:`Zhang2008`.

    Parameters
    ----------
    ci_estimator : Callable
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
    max_iter : int
        The maximum number of iterations through the graph to apply
        orientation rules.
    max_combinations : int, optional
        Maximum number of tries with a conditioning set chosen from the set of possible
        parents still, by default None. If None, then will not be used. If set, then
        the conditioning set will be chosen lexographically based on the sorted
        test statistic values of 'ith Pa(X) -> X', for each possible parent node of 'X'.
    apply_orientations : bool
        Whether or not to apply orientation rules given the learned skeleton graph
        and separating set per pair of variables. If ``True`` (default), will
        apply Zhang's orientation rules R0-10, orienting colliders and certain
        arrowheads and tails :footcite:`Zhang2008`.
    selection_bias : bool
        Whether or not to account for selection bias within the causal PAG.
        See :footcite:`Zhang2008`.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Note that the algorithm is called "fast causal inference", but in reality
    the algorithm is quite expensive in terms of the number of conditional
    independence tests it must run.
    """

    graph_: SemiMarkovianEquivalenceClass
    separating_sets_: Optional[SeparatingSet]

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = None,
        apply_orientations: bool = True,
        selection_bias: bool = False,
        max_path_length: int = None,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator=ci_estimator,
            alpha=alpha,
            min_cond_set_size=min_cond_set_size,
            max_cond_set_size=max_cond_set_size,
            max_combinations=max_combinations,
            apply_orientations=apply_orientations,
            **ci_estimator_kwargs,
        )

        if max_path_length is None:
            max_path_length = np.inf
        self.max_path_length = max_path_length
        self.selection_bias = selection_bias
        self.max_iter = max_iter

    def orient_unshielded_triples(self, graph: SemiMarkovianEquivalenceClass, sep_set: SeparatingSet) -> None:
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : SemiMarkovianEquivalenceClass
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
                    self._orient_collider(graph, v_i, u, v_j)

    def _orient_collider(
        self, graph: SemiMarkovianEquivalenceClass, v_i: Column, u: Column, v_j: Column
    ) -> None:
        logger.info(
            f"orienting collider: {v_i} -> {u} and {v_j} -> {u} to make {v_i} -> {u} <- {v_j}."
        )
        if graph.has_edge(v_i, u, graph.circle_edge_name):
            graph.orient_uncertain_edge(v_i, u)
        if graph.has_edge(v_j, u, graph.circle_edge_name):
            graph.orient_uncertain_edge(v_j, u)

    def _apply_rule1(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column) -> bool:
        """Apply rule 1 of the FCI algorithm.

        If A *-> u o-* C, A and C are not adjacent,
        then we can orient the triple as A *-> u -> C.

        Parameters
        ----------
        graph : SemiMarkovianEquivalenceClass
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False

        # If A *-> u o-* C, A and C are not adjacent,
        # then we can orient the triple as A *-> u -> C.
        # check that a and c are not adjacent
        if a not in graph.neighbors(c):
            # check a *-> u o-* c
            if (
                graph.has_edge(a, u, graph.directed_edge_name) or graph.has_edge(a, u, graph.bidirected_edge_name)
            ) and graph.has_edge(c, u, graph.circle_edge_name):
                logger.info(f"Rule 1: Orienting edge {u} o-* {c} to {u} -> {c}.")
                # orient the edge from u to c and delete
                # the edge from c to u
                if graph.has_edge(u, c, graph.circle_edge_name):
                    graph.orient_uncertain_edge(u, c)
                if graph.has_edge(c, u, graph.circle_edge_name):
                    graph.remove_edge(c, u, graph.circle_edge_name)
                added_arrows = True

        return added_arrows

    def _apply_rule2(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column) -> bool:
        """Apply rule 2 of FCI algorithm.

        If

        - A -> u *-> C, or A *-> u -> C, and
        - A *-o C,

        then orient A *-> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False
        # check that a *-o c edge exists
        if graph.has_edge(a, c, graph.circle_edge_name):
            # - A -> u *-> C, or A *-> u -> C, and
            # - A *-o C,
            # check for A -> u and check that u *-> c
            condition_one = (
                graph.has_edge(a, u, graph.directed_edge_name)
                and not graph.has_edge(u, a, graph.directed_edge_name)
                and not graph.has_edge(u, a, graph.circle_edge_name)
                and (graph.has_edge(u, c, graph.directed_edge_name) or 
                graph.has_edge(u, c, graph.bidirected_edge_name))
            )

            # check that a *-> u -> c
            condition_two = (
                (graph.has_edge(a, u, graph.directed_edge_name) or 
                graph.has_edge(a, u, graph.bidirected_edge_name))
                and graph.has_edge(u, c, graph.directed_edge_name)
                and not graph.has_edge(c, u, graph.directed_edge_name)
                and not graph.has_edge(c, u, graph.circle_edge_name)
            )

            if condition_one or condition_two:
                logger.info(f"Rule 2: Orienting circle edge to {a} -> {c}")
                # orient a *-> c
                graph.orient_uncertain_edge(a, c)
                added_arrows = True
        return added_arrows

    def _apply_rule3(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column) -> bool:
        """Apply rule 3 of FCI algorithm.

        If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
        and v *-o u, then orient v *-> u.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False
        # check that a and c are not adjacent
        if c not in graph.neighbors(a):
            # If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
            # and v *-o u, then orient v *-> u.
            # check that a *-> u <-* c
            condition_one = (graph.has_edge(a, u, graph.directed_edge_name) or graph.has_edge(a, u, graph.bidirected_edge_name)) and (
                graph.has_edge(c, u, graph.directed_edge_name) or graph.has_edge(c, u, graph.bidirected_edge_name)
            )
            if not condition_one:  # add quick check here to skip non-relevant u nodes
                return added_arrows

            # check for all other neighbors to find a 'v' node
            # with the structure A *-o v o-* C
            for v in graph.neighbors(u):
                # check that v is not a, or c
                if v in (a, c):
                    continue

                # check that v *-o u
                if not graph.has_edge(v, u, graph.circle_edge_name):
                    continue

                # check that a *-o v o-* c
                condition_two = graph.has_edge(a, v, graph.circle_edge_name) and graph.has_edge(c, v, graph.circle_edge_name)
                if condition_one and condition_two:
                    logger.info(f"Rule 3: Orienting {v} -> {u}.")
                    graph.orient_uncertain_edge(v, u)
                    added_arrows = True
        return added_arrows

    def _apply_rule4(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column, sep_set) -> Tuple[bool, Dict[Column, None]]:
        """Apply rule 4 of FCI algorithm.

        If a path, U = <v, ..., a, u, c> is a discriminating
        path between v and c for u, u o-* c, u in SepSet(v, c),
        orient u -> c. Else, orient a <-> u <-> c.

        A discriminating path, p, is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        Parameters
        ----------
        graph : PAG
            PAG to orient.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.
        sep_set : set
            The separating set to check.

        Notes
        -----
        ...
        """
        import pywhy_graphs as pgraph

        added_arrows = False
        explored_nodes: Dict[Column, None] = dict()

        # a must point to c for us to begin a discriminating path and
        # not be bi-directional
        if not graph.has_edge(a, c, graph.directed_edge_name) or graph.has_edge(c, a, graph.bidirected_edge_name):
            return added_arrows, explored_nodes

        # c must also point to u with a circle edge
        # check u o-* c
        if not graph.has_edge(c, u, graph.circle_edge_name):
            return added_arrows, explored_nodes

        # 'a' cannot be a definite collider if there is no arrow pointing from
        # u to a either as: u -> a, or u <-> a
        if not graph.has_edge(u, a, graph.directed_edge_name) and not graph.has_edge(u, a, graph.bidirected_edge_name):
            return added_arrows, explored_nodes

        explored_nodes, found_discriminating_path, disc_path = pgraph.discriminating_path(
            graph, u, a, c, self.max_path_length
        )
        disc_path_str = " ".join(
            [
                f'{disc_path[idx]}, {disc_path[idx + 1]}'
                for idx in range(len(disc_path) - 1)
            ]
        )
        if found_discriminating_path:
            last_node = list(explored_nodes.keys())[-1]

            # now check if u is in SepSet(v, c)
            # handle edge case where sep_set is empty.
            if last_node in sep_set:
                if is_in_sep_set(u, sep_set, last_node, c, "any"):  # u in sep_set[last_node][c]:
                    # orient u -> c
                    graph.remove_edge(c, u, graph.circle_edge_name)
                if graph.has_edge(u, c, graph.circle_edge_name):
                    graph.orient_uncertain_edge(u, c)
                logger.info(f"Rule 4: orienting {u} -> {c}.")
                logger.info(disc_path_str)
            else:
                # orient u <-> c
                if graph.has_edge(u, c, graph.circle_edge_name):
                    graph.orient_uncertain_edge(u, c)
                if graph.has_edge(c, u, graph.circle_edge_name):
                    graph.orient_uncertain_edge(c, u)
                logger.info(f"Rule 4: orienting {u} <-> {c}.")
                logger.info(disc_path_str)
            added_arrows = True

        return added_arrows, explored_nodes

    def _apply_rule8(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column) -> bool:
        """Apply rule 8 of FCI algorithm.

        If A -> u -> C, or A -o B -> C
        and A o-> C, then orient A o-> C as A -> C.

        Without dealing with selection bias, we ignore
        A -o B -> C condition.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        # If A -> u -> C,
        # and A o-> C, then orient A o-> C as A -> C.
        added_arrows = False

        # First check that A o-> C
        if graph.has_edge(c, a, graph.circle_edge_name) and graph.has_edge(a, c, graph.directed_edge_name):
            # check that A -> u -> C
            condition_one = graph.has_edge(a, u, graph.directed_edge_name) and not graph.has_edge(u, a, graph.circle_edge_name)
            condition_two = graph.has_edge(u, c, graph.directed_edge_name) and not graph.has_edge(c, u, graph.circle_edge_name)

            if condition_one and condition_two:
                logger.info(f"Rule 8: Orienting {a} o-> {c} as {a} -> {c}.")
                # now orient A o-> C as A -> C
                if graph.has_edge(c, a, graph.circle_edge_name):
                    graph.remove_edge(c, a, graph.circle_edge_name)
                    added_arrows = True
        return added_arrows

    def _apply_rule9(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column) -> Tuple[bool, List]:
        """Apply rule 9 of FCI algorithm.

        If A o-> C and p = <A, u, v, ..., C> is an uncovered
        possibly directed path from A to C such that u and C
        are not adjacent, orient A o-> C  as A -> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        uncov_path : list
            The uncovered potentially directed path from 'a' to 'c' through 'u'.
        """
        import pywhy_graphs as pgraph

        added_arrows = False
        uncov_path: List[Column] = []

        # Check A o-> C and # check that u is not adjacent to c
        if (graph.has_edge(c, a, graph.circle_edge_name) and graph.has_edge(a, c, graph.directed_edge_name)) and c not in graph.neighbors(u):
            # check that <a, u> is potentially directed
            if graph.has_edge(a, u):
                # check that A - u - v, ..., c is an uncovered pd path
                uncov_path, path_exists = pgraph.uncovered_pd_path(
                    graph, u, c, max_path_length=self.max_path_length, first_node=a
                )

                # orient A o-> C to A -> C
                if path_exists:
                    logger.info(f"Rule 9: Orienting edge {a} o-> {c} to {a} -> {c}.")
                    if graph.has_edge(c, a, graph.circle_edge_name):
                        graph.remove_edge(c, a, graph.circle_edge_name)
                        added_arrows = True

        return added_arrows, uncov_path

    def _apply_rule10(self, graph: SemiMarkovianEquivalenceClass, u: Column, a: Column, c: Column) -> Tuple[bool, List, List]:
        """Apply rule 10 of FCI algorithm.

        If A o-> C and u -> C <- v and

        - p1 is an uncovered pd path from A to u
        - p2 is an uncovered pd path from A to v

        Then say m is adjacent to A on p1 (could be u).
        Say w is adjacent to A on p2 (could be v).

        If m and w are distinct and not adjacent, then
        orient A o-> C  as A -> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        import pywhy_graphs as pgraph

        added_arrows = False
        a_to_u_path: List[Column] = []
        a_to_v_path: List[Column] = []

        # Check A o-> C
        if graph.has_edge(c, a, graph.circle_edge_name) and graph.has_edge(a, c, graph.directed_edge_name):
            # check that u -> C
            if graph.has_edge(u, c, graph.directed_edge_name) and not graph.has_edge(c, u, graph.circle_edge_name):
                # loop through all adjacent neighbors of c now to get
                # possible 'v' node
                for v in graph.neighbors(c):
                    if v in (a, u):
                        continue

                    # make sure v -> C and not v o-> C
                    if not graph.has_edge(v, c, graph.directed_edge_name) or graph.has_edge(c, v, graph.circle_edge_name):
                        continue

                    # At this point, we want the paths from A to u and A to v
                    # to begin with a distinct m and w node, else we will not
                    # apply R10. Thus, we will get all 2-pairs of neighbors of A
                    # that:
                    # i) begin the uncovered pd path and
                    # ii) are distinct (done by construction) here
                    for (m, w) in combinations(graph.neighbors(a), 2):  # type: ignore
                        if m == c or w == c:
                            continue

                        # m and w must be on a potentially directed path
                        if not graph.has_edge(a, m, graph.directed_edge_name) or not graph.has_edge(a, w, graph.directed_edge_name):
                            continue

                        # we do not know which path a-u or a-v, m and w are on
                        # so we must traverse the graph in both directions
                        # get the uncovered pd path from A to u just once
                        found_uncovered_a_to_v = False
                        a_to_u_path, found_uncovered_a_to_u = pgraph.uncovered_pd_path(
                            graph, a, u, max_path_length=self.max_path_length, second_node=m
                        )

                        # we did not find a path from 'a' to 'u' through 'm', so look for
                        # a path through 'w' instead
                        if not found_uncovered_a_to_u:
                            a_to_u_path, found_uncovered_a_to_u = pgraph.uncovered_pd_path(
                                graph, a, u, max_path_length=self.max_path_length, second_node=w
                            )
                            # if we don't have an uncovered pd path here, then no point in looking
                            # for other paths
                            if found_uncovered_a_to_u:
                                a_to_v_path, found_uncovered_a_to_v = pgraph.uncovered_pd_path(
                                    graph, a, v, max_path_length=self.max_path_length, second_node=m
                                )
                        else:
                            a_to_v_path, found_uncovered_a_to_v = pgraph.uncovered_pd_path(
                                graph, a, v, max_path_length=self.max_path_length, second_node=w
                            )

                        # if we have not found another path, then just continue
                        if not found_uncovered_a_to_v:
                            continue

                        # at this point, we have an uncovered path from a to u and a to v
                        # with a distinct second node on both paths
                        # orient A o-> C to A -> C
                        logger.info(f"Rule 10: Orienting edge {a} o-> {c} to {a} -> {c}.")
                        if graph.has_edge(c, a, graph.circle_edge_name):
                            graph.remove_edge(c, a, graph.circle_edge_name)
                            added_arrows = True

        return added_arrows, a_to_u_path, a_to_v_path

    def _apply_rules_1to10(self, graph: SemiMarkovianEquivalenceClass, sep_set: Dict[str, Dict[str, Set[Any]]]):
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:
            change_flag = False
            logger.info(f"Running R1-10 for iteration {idx}")

            for u in graph.nodes:
                for (a, c) in permutations(graph.neighbors(u), 2):
                    # apply R1-3 to orient triples and arrowheads
                    r1_add = self._apply_rule1(graph, u, a, c)
                    r2_add = self._apply_rule2(graph, u, a, c)
                    r3_add = self._apply_rule3(graph, u, a, c)

                    # apply R4, orienting discriminating paths
                    r4_add, _ = self._apply_rule4(graph, u, a, c, sep_set)

                    # apply R8 to orient more tails
                    r8_add = self._apply_rule8(graph, u, a, c)

                    # apply R9-10 to orient uncovered potentially directed paths
                    r9_add, _ = self._apply_rule9(graph, u, a, c)

                    # a and c are neighbors of u, so u is the endpoint desired
                    r10_add, _, _ = self._apply_rule10(graph, a, c, u)

                    # see if there was a change flag
                    if (
                        any([r1_add, r2_add, r3_add, r4_add, r8_add, r9_add, r10_add])
                        and not change_flag
                    ):
                        logger.info(
                            f"{change_flag} with {[r1_add, r2_add, r3_add, r4_add, r8_add, r9_add, r10_add]}"
                        )
                        change_flag = True

            # check if we should continue or not
            if not change_flag:
                finished = True
                logger.info(f"Finished applying R1-4, and R8-10 with {idx} iterations")
                break
            idx += 1

    def _learn_better_skeleton(
        self,
        X: pd.DataFrame,
        pag: nx.Graph,
        sep_set: SeparatingSet,
        fixed_edges: Optional[Set] = set(),
    ) -> Tuple[nx.Graph, SeparatingSet]:
        from causal_networkx.discovery.skeleton import learn_skeleton_graph_with_pdsep

        # convert the adjacency graph
        adj_graph = pag.to_adjacency_graph()

        # perform pairwise tests to learn skeleton
        # skel_alg = LearnSkeleton(
        #     self.ci_estimator,
        #     adj_graph=adj_graph,
        #     sep_set=sep_set,
        #     fixed_edges=fixed_edges,
        #     alpha=self.alpha,
        #     min_cond_set_size=1,
        #     max_cond_set_size=self.max_cond_set_size,
        #     skeleton_method="pds",
        #     max_path_length=self.max_path_length,
        #     pag=pag,
        #     keep_sorted=False,
        #     **self.ci_estimator_kwargs,
        # )
        # skel_alg.fit(X)
        # skel_graph = skel_alg.adj_graph_
        # sep_set = skel_alg.sep_set_
        # perform pairwise tests to learn skeleton
        skel_graph, sep_set = learn_skeleton_graph_with_pdsep(
            X,
            self.ci_estimator,
            adj_graph=adj_graph,
            sep_set=sep_set,
            fixed_edges=fixed_edges,
            alpha=self.alpha,
            min_cond_set_size=1,
            max_cond_set_size=self.max_cond_set_size,
            max_path_length=self.max_path_length,
            pag=pag,
            **self.ci_estimator_kwargs,
        )
        return skel_graph, sep_set

    def learn_skeleton(
        self,
        X: pd.DataFrame,
        graph: nx.Graph = None,
        sep_set: Optional[SeparatingSet] = None,
        fixed_edges: Optional[Set] = None,
    ) -> Tuple[
        nx.Graph,
        SeparatingSet,
    ]:
        """Learn skeleton from data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.
        graph : nx.Graph
            The undirected graph containing initialized skeleton of the causal
            relationships.
        sep_set : set
            The separating set.
        fixed_edges : set, optional
            The set of fixed edges. By default, is the empty set.

        Returns
        -------
        pag : PAG
            The skeleton graph.
        sep_set : Dict[str, Dict[str, Set]]
            The separating set.
        """
        import pywhy_graphs as pgraph

        nodes = X.columns.values

        # initialize graph object to apply learning
        graph, sep_set = self._initialize_graph(nodes)

        # learn the initial skeleton of the graph
        skel_graph, sep_set = super().learn_skeleton(
            X, graph, sep_set, fixed_edges
        )

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = pgraph.PAG(incoming_uncertain_data=skel_graph, name="PAG derived with FCI")

        # orient colliders
        self.orient_unshielded_triples(pag, sep_set)

        # # now compute all possibly d-separating sets and learn a better skeleton
        skel_graph, sep_set = self._learn_better_skeleton(X, pag, sep_set, fixed_edges)

        self.skel_graph_ = skel_graph.copy()
        return skel_graph, sep_set

    def orient_edges(self, graph: SemiMarkovianEquivalenceClass, sep_set: SeparatingSet) -> SemiMarkovianEquivalenceClass:
        # orient colliders again
        self._orient_unshielded_triples(graph, sep_set)
        self.orient_coll_graph = graph.copy()

        # run the rest of the rules to orient as many edges
        # as possible
        self._apply_rules_1to10(graph, sep_set)
        return graph

    def convert_skeleton_graph(self, graph: nx.Graph) -> SemiMarkovianEquivalenceClass:
        import pywhy_graphs as pgraph

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = pgraph.PAG(incoming_uncertain_data=graph, name="PAG derived with FCI")
        return pag
