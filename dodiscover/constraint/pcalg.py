import logging
from collections import defaultdict
from copy import copy
from itertools import combinations, permutations
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pywhy_graphs import CPDAG

from dodiscover.ci.base import BaseConditionalIndependenceTest
from dodiscover.constraint.utils import is_in_sep_set

from ._classes import BaseConstraintDiscovery
from ._protocol import EquivalenceClassProtocol

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
        apply Meek's orientation rules R0-3, orienting colliders and certain
        arrowheads :footcite:`Meek1995`.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    graph_ : CPDAG
        The graph discovered.
    separating_sets_ : dict
        The dictionary of separating sets, where it is a nested dictionary from
        the variable name to the variable it is being compared to the set of
        variables in the graph that separate the two.

    References
    ----------
    .. footbibliography::
    """

    graph_: EquivalenceClassProtocol
    separating_sets_: Dict[str, Dict[str, Set[Any]]]

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = None,
        apply_orientations: bool = True,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size=min_cond_set_size,
            max_cond_set_size=max_cond_set_size,
            max_combinations=max_combinations,
            **ci_estimator_kwargs,
        )
        self.max_iter = max_iter
        self.apply_orientations = apply_orientations

    def convert_skeleton_graph(self, graph: nx.Graph) -> CPDAG:
        """Convert skeleton graph as undirected networkx Graph to CPDAG.

        Parameters
        ----------
        graph : nx.Graph
            Converts a skeleton graph to the representation needed
            for PC algorithm, a CPDAG.

        Returns
        -------
        graph : CPDAG
            The CPDAG class.
        """
        # convert Graph object to a CPDAG object with
        # all undirected edges
        graph = CPDAG(incoming_uncertain_data=graph)
        return graph

    def orient_edges(
        self, skel_graph: CPDAG, sep_set: Dict[str, Dict[str, List[Set[Any]]]]
    ) -> CPDAG:
        """Orient edges in a skeleton graph to estimate the causal DAG, or CPDAG.

        Uses the separation sets to orient edges via conditional independence
        testing. These are known as the Meek rules :footcite:`Meek1995`.

        Parameters
        ----------
        skel_graph : causal_networkx.CPDAG
            A skeleton graph. If ``None``, then will initialize PC using a
            complete graph. By default None.
        sep_set : Dict[Dict[Set[Set[Any]]]]
            The separating set between any two nodes.
        """
        node_ids = skel_graph.nodes()

        # for all pairs of non-adjacent variables with a common neighbor
        # check if we can orient the edge as a collider
        self._orient_unshielded_triples(skel_graph, sep_set)

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
                r1_add = self._apply_meek_rule1(skel_graph, i, j)

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                r2_add = self._apply_meek_rule2(skel_graph, i, j)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                r3_add = self._apply_meek_rule3(skel_graph, i, j)

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

        return skel_graph

    def _orient_unshielded_triples(
        self, graph: CPDAG, sep_set: Dict[str, Dict[str, List[Set[Any]]]]
    ) -> None:
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : CPDAG
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

    def _orient_collider(self, graph: CPDAG, v_i, u, v_j) -> None:
        logger.info(
            f"orienting collider: {v_i} -> {u} and {v_j} -> {u} to make {v_i} -> {u} <- {v_j}."
        )

        if graph.has_edge(v_i, u, graph.undirected_edge_name):
            graph.orient_uncertain_edge(v_i, u)
        if graph.has_edge(v_j, u, graph.undirected_edge_name):
            graph.orient_uncertain_edge(v_j, u)

    def _apply_meek_rule1(self, graph: CPDAG, i, j) -> bool:
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
                if frozenset(k, i, j) in graph.excluded_triples:
                    continue

                # Make i-j into i->j
                logger.info(f"R1: Removing edge ({i}, {j}) and orienting as {k} -> {i} -> {j}.")
                graph.orient_uncertain_edge(i, j)

                added_arrows = True
                break
        return added_arrows

    def _apply_meek_rule2(self, graph: CPDAG, i, j) -> bool:
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
                    if frozenset(i, k, j) in graph.excluded_triples:
                        candidate_k.remove(k)

            # if there are candidate 'k' nodes, then orient the edge accordingly
            if len(candidate_k) > 0:
                # Make i-j into i->j
                logger.info(f"R2: Removing edge {i}-{j} to form {i}->{j}.")
                graph.orient_uncertain_edge(i, j)
                added_arrows = True
        return added_arrows

    def _apply_meek_rule3(self, graph: CPDAG, i, j) -> bool:
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
                if frozenset(l, i, k) in graph.excluded_triples:
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


class ConservativeVotingPC(PC):
    """Conservative/MajorityVote-PC algorithm with causal discovery.

    Assumes causal sufficiency, that is, all confounders in the
    causal graph are observed variables. See :footcite:`ramsey2012adjacency` for
    full details on the algorithm for conservative orientation. See :footcite:`Colombo2012_MPC`
    for full details on the algorithm for Majority Voting orientation.

    Parameters
    ----------
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    init_graph : nx.Graph | ADMG, optional
        An initialized graph. If ``None``, then will initialize PC using a
        complete graph. By default None.
    fixed_edges : nx.Graph, optional
        An undirected graph with fixed edges. If ``None``, then will initialize PC using a
        complete graph. By default None.
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
        apply Meek's orientation rules R0-3, orienting colliders and certain
        arrowheads :footcite:`Meek1995`.
    vote_threshold : float
        The voting threshold to orient a triple collider. By default, is None for
        pure conservative PC. See Notes for details on how to set value for voting.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    graph_ : CPDAG
        The graph discovered.
    separating_sets_ : dict
        The dictionary of separating sets, where it is a nested dictionary from
        the variable name to the variable it is being compared to the set of
        variables in the graph that separate the two.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    The majority voting scheme will set a triple as either a definite collider,
    definite non-collider, or mark it as an unfaithful triple. If the majority voting
    `vote_threshold` is set to 0.5, then it will:

    - mark definite collider if < 0.5 of tests say X || Y has Z
    - mark definite non-collider if > 0.5 of tests say X || Y has Z
    - mark unfaithful if exactly 0.5 of tests say X || Y has Z

    If we modify the `vote_threshold` to 0.3, then it will:

    - mark definite collider if < 0.3 of tests say X || Y has Z
    - mark definite non-collider if > 0.7 of tests say X || Y has Z
    - mark unfaithful if 0.3 to 0.7 of tests say X || Y has Z
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, CPDAG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = None,
        apply_orientations: bool = True,
        vote_threshold: float = None,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size,
            max_cond_set_size,
            max_iter,
            max_combinations,
            apply_orientations,
            **ci_estimator_kwargs,
        )
        self.vote_threshold = vote_threshold

    def convert_skeleton_graph(self, graph: nx.Graph) -> CPDAG:
        return super().convert_skeleton_graph(graph)

    def _orient_unshielded_triples(
        self, graph: CPDAG, sep_set: Dict[str, Dict[str, List[Set[Any]]]]
    ):
        """Orient unshielded triples conservatively."""

        # for every node in the PAG, evaluate neighbors that have any edge
        for u in graph.nodes:
            for v_i, v_j in combinations(graph.adjacencies(u), 2):
                # Check that there is no edge of any type between
                # v_i and v_j, else this is a "shielded" collider.
                if not graph.has_adjacency(v_i, v_j):
                    # check the triple
                    self._check_triple(graph, self.X, v_i, u, v_j, sep_set)

    def _check_triple(self, graph: ExtendedPattern, data, v_i, u, v_j, sep_set):
        """Check the triple (v_i, u, v_j) using the conservative rules.

        Parameters
        ----------
        graph : CPDAG
            _description_
        data : _type_
            _description_
        v_i : _type_
            _description_
        u : _type_
            _description_
        v_j : _type_
            _description_
        sep_set : _type_
            _description_
        """
        # now collect all potential parents of 'v_i' and 'v_j'
        potential_parent_set_i = set(graph.adjacencies(v_i))
        potential_parent_set_j = set(graph.adjacencies(v_j))
        potential_parent_set = potential_parent_set_i.union(potential_parent_set_j)

        # now re-test all subsets of the potential parents
        conservative_sep_sets = []
        for idx in range(1, len(potential_parent_set) + 1):
            for pparent_subset in combinations(potential_parent_set, idx):
                # now check all subsets of the parents to determine if 'u' is in
                # the separating set
                _, pvalue = self.ci_estimator.test(data, v_i, v_j, pparent_subset)

                # independent...
                if pvalue > self.alpha:
                    conservative_sep_sets.append(set(pparent_subset))

        # if 'u' is NOT in ANY separating set of the conservative check
        if all(u not in sep_ for sep_ in conservative_sep_sets):
            # then it is a collider, so add to the separating set if not there
            # and orient the collider
            if not is_in_sep_set(u, sep_set, v_i, v_j, mode="any"):
                sep_set[v_i][v_j] |= (u,)
                sep_set[v_j][v_i] |= (u,)
            self._orient_collider(graph, v_i, u, v_j)

        # if 'u' is in SOME of the separating sets of the conservative check, but not ALL
        elif any(u in sep_ for sep_ in conservative_sep_sets) and not all(
            u in sep_ for sep_ in conservative_sep_sets
        ):
            # it is unfaithful triple
            self._mark_unfaithful_triple(graph, v_i, u, v_j)

    def _mark_unfaithful_triple(self, graph: ExtendedPattern, v_i, u, v_j):
        """Mark an unshielded triple as unfaithful.

        Parameters
        ----------
        graph : _type_
            _description_
        v_i : node
            First node.
        u : node
            Second node.
        v_j : node
            Third node.
        """
        graph.mark_unfaithful_triple(v_i, u, v_j)


class RobustPC(PC):
    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, DAG, CPDAG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = None,
        apply_orientations: bool = True,
        mci_alpha: float = 0.05,
        max_conds_x: int = None,
        max_conds_y: int = None,
        size_inclusive: bool = False,
        mci_ci_estimator: Callable = None,
        partial_knowledge: object = None,
        only_mci: bool = False,
        use_children: bool = False,
        use_parents: bool = True,  # TODO: remove cuz temporary
        skip_first_stage: bool = False,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size,
            max_cond_set_size,
            max_iter,
            max_combinations,
            apply_orientations,
            **ci_estimator_kwargs,
        )
        self.max_conds_x = max_conds_x
        self.max_conds_y = max_conds_y
        self.size_inclusive = size_inclusive
        self.mci_alpha = mci_alpha
        self.partial_knowledge = partial_knowledge
        self.only_mci = only_mci
        if mci_ci_estimator is None:
            self.mci_ci_estimator = ci_estimator

        self.use_children = use_children
        self.skip_first_stage = skip_first_stage
        self.use_parents = use_parents

    def _learn_first_phase(self, X, graph, sep_set, fixed_edges):
        # learn skeleton using original PC algorithm
        graph, sep_set, test_stat_dict, pvalue_dict = super().learn_skeleton(
            X, graph, sep_set, fixed_edges
        )
        # convert graph to a CPDAG
        graph = self.convert_skeleton_graph(graph)
        # orient the edges of the skeleton graph to build up a set of
        # "definite" parents
        graph = self.orient_edges(graph, sep_set)
        return graph, sep_set, test_stat_dict, pvalue_dict

    def learn_skeleton(
        self,
        X: pd.DataFrame,
        graph: nx.Graph = None,
        sep_set: Dict[str, Dict[str, Set[Set[Any]]]] = None,
        fixed_edges: Set = None,
    ) -> Tuple[
        nx.Graph,
        Dict[str, Dict[str, Set[Any]]],
        Dict[Any, Dict[Any, float]],
        Dict[Any, Dict[Any, float]],
    ]:
        from causal_networkx.discovery import LearnSkeleton

        if graph is None:
            nodes = X.columns
            graph = nx.complete_graph(nodes, create_using=nx.Graph)
        if sep_set is None:
            # keep track of separating sets
            sep_set = defaultdict(lambda: defaultdict(set(set)))
        orig_graph = graph.copy()
        orig_sep_set = sep_set.copy()
        test_stat_dict = dict()
        pvalue_dict = dict()

        if not self.skip_first_stage:
            graph, sep_set, test_stat_dict, pvalue_dict = self._learn_first_phase(
                X, graph, sep_set, fixed_edges
            )

        # store the estimated "definite" parents/children for each node
        def_parent_dict = dict()
        def_children_dict = dict()

        # now obtain the definite parents for every node in the set either using
        # partial knowledge oracle, or using the existing graph oriented after initial PC
        if hasattr(self.partial_knowledge, "get_parents"):
            test_stat_dict = dict()
            for node in graph.nodes:
                parents = self.partial_knowledge.get_parents(node)  # type: ignore
                children = self.partial_knowledge.get_children(node)  # type: ignore
                def_parent_dict[node] = parents
                def_children_dict[node] = children

                test_stat_dict[node] = {_node: np.inf for _node in parents}

                if self.use_children:
                    test_stat_dict[node] = {_node: np.inf for _node in children}
        else:
            # use the estimated parents/children
            def_parent_dict = self._estimate_definite_parents(graph)
            def_children_dict = self._estimate_definite_children(graph)

            # use definite parents to filter the dependencies in the test statistics / pvalue
            # removing them from the possible adjacency list
            for node, parents in def_parent_dict.items():
                # create a copy of the possible parents, since we will be removing
                # certain keys in the nested dictionary
                possible_parents = copy(list(test_stat_dict[node].keys()))
                children = def_children_dict[node]
                for adj_node in possible_parents:
                    # now remove any adjacent nodes from consideration if they
                    # are not part of parent set
                    check_condition = True
                    if self.use_parents:
                        check_condition = adj_node not in parents

                    # optionally, also include children
                    if self.use_children:
                        check_condition = check_condition and (adj_node not in children)

                    if check_condition:
                        test_stat_dict[node].pop(adj_node)
                        # pvalue_dict[node].pop(parent)

        self._inter_test_stat_dict = test_stat_dict
        self.def_parents_ = def_parent_dict
        self.def_children_ = def_children_dict

        # now we will re-learn the skeleton using the MCI condition
        skel_alg = LearnSkeleton(
            self.mci_ci_estimator,
            adj_graph=orig_graph,
            sep_set=orig_sep_set,
            fixed_edges=fixed_edges,
            alpha=self.mci_alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=1,
            keep_sorted=False,
            with_mci=True,
            max_conds_x=self.max_conds_x,
            max_conds_y=self.max_conds_y,
            parent_dep_dict=test_stat_dict,
            size_inclusive=self.size_inclusive,
            only_mci=self.only_mci,
            multicomp_method="holm",
            **self.ci_estimator_kwargs,
        )
        skel_alg.fit(X)

        skel_graph = skel_alg.adj_graph_
        sep_set = skel_alg.sep_set_
        test_stat_dict = skel_alg.test_stat_dict_
        pvalue_dict = skel_alg.pvalue_dict_

        return skel_graph, sep_set, test_stat_dict, pvalue_dict

    def _estimate_definite_parents(self, graph: CPDAG):
        def_parent_dict = dict()

        for node in graph.nodes:
            # get all predecessors of the node that have an arrowhead into node
            def_parent_dict[node] = list(graph.predecessors(node))

        return def_parent_dict

    def _estimate_definite_children(self, graph: CPDAG):
        def_children_dict = dict()

        for node in graph.nodes:
            # get all predecessors of the node that have an arrowhead into node
            def_children_dict[node] = list(graph.successors(node))

        return def_children_dict
