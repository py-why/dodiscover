import logging
from itertools import combinations, permutations
from typing import Optional

import networkx as nx

from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.constraint.config import SkeletonMethods
from dodiscover.constraint.utils import is_in_sep_set
from dodiscover.typing import Column, SeparatingSet

from .._protocol import EquivalenceClass
from ._classes import BaseConstraintDiscovery

logger = logging.getLogger()


class PC(BaseConstraintDiscovery):
    """Peter and Clarke (PC) algorithm for causal discovery.

    Assumes causal sufficiency, that is, all confounders in the
    causal graph are observed variables. See :footcite:`Spirtes1993` for
    full details on the algorithm.

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
        apply Meek's orientation rules R0-3, orienting colliders and certain
        arrowheads :footcite:`Meek1995`.
    max_iter : int
        The maximum number of iterations through the graph to apply
        orientation rules.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    graph_ : EquivalenceClass
        The equivalence class of graphs discovered.
    separating_sets_ : dict of dict of list of set
        The dictionary of separating sets, where it is a nested dictionary from
        the variable name to the variable it is being compared to the set of
        variables in the graph that separate the two.

    References
    ----------
    .. footbibliography::
    """

    graph_: EquivalenceClass
    separating_sets_: SeparatingSet

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
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size=min_cond_set_size,
            max_cond_set_size=max_cond_set_size,
            max_combinations=max_combinations,
            skeleton_method=skeleton_method,
            **ci_estimator_kwargs,
        )
        self.max_iter = max_iter
        self.apply_orientations = apply_orientations

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClass:
        """Convert skeleton graph as undirected networkx Graph to CPDAG.

        Parameters
        ----------
        graph : nx.Graph
            Converts a skeleton graph to the representation needed
            for PC algorithm, a CPDAG.

        Returns
        -------
        graph : EquivalenceClass
            The CPDAG class.
        """
        from pywhy_graphs import CPDAG

        # convert Graph object to a CPDAG object with
        # all undirected edges
        graph = CPDAG(incoming_undirected_edges=graph)
        return graph

    def orient_edges(self, graph: EquivalenceClass) -> None:
        """Orient edges in a skeleton graph to estimate the causal DAG, or CPDAG.

        These are known as the Meek rules :footcite:`Meek1995`. They are deterministic
        in the sense that they are logical characterizations of what edges must be
        present given the rest of the local graph structure.

        Parameters
        ----------
        graph : EquivalenceClass
            A skeleton graph. If ``None``, then will initialize PC using a
            complete graph. By default None.
        """
        node_ids = graph.nodes

        # For all the combination of nodes i and j, apply the following
        # rules.
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:  # type: ignore
            change_flag = False
            for (i, j) in permutations(node_ids, 2):
                if i == j:
                    continue
                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                r1_add = self._apply_meek_rule1(graph, i, j)

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                r2_add = self._apply_meek_rule2(graph, i, j)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                r3_add = self._apply_meek_rule3(graph, i, j)

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                # However, this rule is not necessary when the PC-algorithm
                # is used to estimate a DAG.

                if any([r1_add, r2_add, r3_add]) and not change_flag:
                    change_flag = True
            if not change_flag:
                finished = True
                logger.info(f"Finished applying R1-3, with {idx} iterations")
                break
            idx += 1

    def orient_unshielded_triples(
        self,
        graph: EquivalenceClass,
        sep_set: SeparatingSet,
    ) -> None:
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : EquivalenceClass
            The CPDAG.
        sep_set : Dict[Dict[Set[Set[Any]]]]
            The separating set between any two nodes.
        """
        # for every node in the PAG, evaluate neighbors that have any edge
        for u in graph.nodes:
            for v_i, v_j in combinations(graph.neighbors(u), 2):
                # Check that there is no edge of any type between
                # v_i and v_j, else this is a "shielded" collider.
                # Then check to see if 'u' is in "any" separating
                # set. If it is not, then there is a collider.
                if v_j not in graph.neighbors(v_i) and not is_in_sep_set(
                    u, sep_set, v_i, v_j, mode="any"
                ):
                    self._orient_collider(graph, v_i, u, v_j)

    def _orient_collider(
        self, graph: EquivalenceClass, v_i: Column, u: Column, v_j: Column
    ) -> None:
        logger.info(
            f"orienting collider: {v_i} -> {u} and {v_j} -> {u} to make {v_i} -> {u} <- {v_j}."
        )

        if graph.has_edge(v_i, u, graph.undirected_edge_name):
            graph.orient_uncertain_edge(v_i, u)
        if graph.has_edge(v_j, u, graph.undirected_edge_name):
            graph.orient_uncertain_edge(v_j, u)

    def _apply_meek_rule1(self, graph: EquivalenceClass, i: Column, j: Column) -> bool:
        """Apply rule 1 of Meek's rules.

        Looks for i - j such that k -> i, such that (k,i,j)
        is an unshielded triple. Then can orient i - j as i -> j.
        """
        added_arrows = False

        # Check if i-j.
        if graph.has_edge(i, j, graph.undirected_edge_name):
            for k in graph.predecessors(i):
                # Skip if k and j are adjacent because then it is a
                # shielded triple
                if j in graph.neighbors(k):
                    continue

                # check if the triple is in the graph's excluded triples
                if frozenset((k, i, j)) in graph.excluded_triples:
                    continue

                # Make i-j into i->j
                logger.info(f"R1: Removing edge ({i}, {j}) and orienting as {k} -> {i} -> {j}.")
                graph.orient_uncertain_edge(i, j)

                added_arrows = True
                break
        return added_arrows

    def _apply_meek_rule2(self, graph: EquivalenceClass, i: Column, j: Column) -> bool:
        """Apply rule 2 of Meek's rules.

        Check for i - j, and then looks for i -> k -> j
        triple, to orient i - j as i -> j.
        """
        added_arrows = False

        # Check if i-j.
        if graph.has_edge(i, j, graph.undirected_edge_name):
            # Find nodes k where k is i->k
            succs_i = set()
            for k in graph.successors(i):
                if not graph.has_edge(k, i, graph.directed_edge_name):
                    succs_i.add(k)
            # Find nodes j where j is k->j.
            preds_j = set()
            for k in graph.predecessors(j):
                if not graph.has_edge(j, k, graph.directed_edge_name):
                    preds_j.add(k)

            # Check if there is any node k where i->k->j.
            candidate_k = succs_i.intersection(preds_j)
            # if the graph has excluded triples, we would check at this point
            if graph.excluded_triples:
                # check if the triple is in the graph's excluded triples
                # if so, remove them from the candidates
                for k in candidate_k:
                    if frozenset((i, k, j)) in graph.excluded_triples:
                        candidate_k.remove(k)

            # if there are candidate 'k' nodes, then orient the edge accordingly
            if len(candidate_k) > 0:
                # Make i-j into i->j
                logger.info(f"R2: Removing edge {i}-{j} to form {i}->{j}.")
                graph.orient_uncertain_edge(i, j)
                added_arrows = True
        return added_arrows

    def _apply_meek_rule3(self, graph: EquivalenceClass, i: Column, j: Column) -> bool:
        """Apply rule 3 of Meek's rules.

        Check for i - j, and then looks for k -> j <- l
        collider, and i - k and i - l, then orient i -> j.
        """
        added_arrows = False

        # Check if i-j first
        if graph.has_edge(i, j, graph.undirected_edge_name):
            # For all the pairs of nodes adjacent to i,
            # look for (k, l), such that j -> l and k -> l
            for (k, l) in combinations(graph.neighbors(i), 2):
                # Skip if k and l are adjacent.
                if l in graph.neighbors(k):
                    continue
                # Skip if not k->j.
                if graph.has_edge(j, k, graph.directed_edge_name) or (
                    not graph.has_edge(k, j, graph.directed_edge_name)
                ):
                    continue
                # Skip if not l->j.
                if graph.has_edge(j, l, graph.directed_edge_name) or (
                    not graph.has_edge(l, j, graph.directed_edge_name)
                ):
                    continue

                # check if the triple is inside graph's excluded triples
                if frozenset((l, i, k)) in graph.excluded_triples:
                    continue

                # if i - k and i - l, then  at this point, we have a valid path
                # to orient
                if graph.has_edge(k, i, graph.undirected_edge_name) and graph.has_edge(
                    l, i, graph.undirected_edge_name
                ):
                    logger.info(f"R3: Removing edge {i}-{j} to form {i}->{j}")
                    graph.orient_uncertain_edge(i, j)
                    added_arrows = True
                    break
        return added_arrows
