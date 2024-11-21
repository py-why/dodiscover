from typing import Callable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pywhy_graphs.algorithms import all_semi_directed_paths, pdag_to_cpdag

from dodiscover.context import Context

from .score_function import ScoreFunction
from .utils import is_clique, powerset, undir_nbrs

# threshold for score changing
THRESHOLD = 1.0e-6


def _score_insert_operators(origin, target, data, context: Context, score_cache: ScoreFunction):
    """Scores a valid insert operation of ``origin -> target``.

    A valid insert operator is described in Theorem 15 of :footcite:`chickering2002optimal`.
    There are two conditions that must be satisfied:

    - $NA_{X, Y} \cup T$ forms a clique.
    - Every semi-directed path from origin to target is blocked by a node from $NA_{X, Y} \cup T$.

    Here, $NA_{X, Y}$ is the set of nodes that are both undirected neighbors to X and
    adjacent to Y.

    $T$ is the set of nodes that are undirected neighbors with target that are not adjacent
    to origin.

    Parameters
    ----------
    origin : Node
        Origin node.
    target : Node
        Target node.
    data : pd.DataFrame
        The dataset.
    context : Context
        Causal context.
    score_cache : Callable
        The scoring function, which takes a dataset, the origin node and target node,
        and returns a score.

    Returns
    -------
    scores : list
        List of delta scores for each valid operator.
    valid_operators : list
        List of valid operators, where each operator is a tuple of the form
        ``(origin, target, T)``, where ``origin`` is the origin node, ``target``
        is the target node, and ``T`` is the set of nodes that are adjacent to target that
        will be oriented pointing towards target.
    """
    G = context.init_graph

    valid_operators = []
    scores = []

    # get the target node's undirected nbrs that are not adjacent to origin
    T0 = undir_nbrs(G, target).difference(G.neighbors(origin))

    # get the undirected nbrs of both origin and adjacent to target
    na_xy = undir_nbrs(G, origin).intersection(G.neighbors(target))

    # get all semi-directed paths from origin to target
    paths = list(all_semi_directed_paths(G, origin, target))

    # define a cache that fast-checks cliques and fast-checks
    # subset of T that block the semi-directed paths from origin to target
    nonclique_cache: Set = set()
    blocked_semi_path: Set = set()

    for T in powerset(T0):
        T = frozenset(T)

        # any superset of a known nonclique is also a nonclique
        if any(T.issuperset(nonclique) for nonclique in nonclique_cache):
            continue

        # condition 1: na_xy union target_nbrs forms a clique
        cond_one = is_clique(G, na_xy.union(T))
        if not cond_one:
            # remove any subset of target_nbrs that contains T as a subset
            # since the clique condition will also not hold for those
            nonclique_cache.add(T)

        # condition 2: every semi-directed path from origin to target is
        # blocked by a node from na_xy union target_nbrs

        # if any existing known blocked path is a superset of T, then it is for sure blocked
        if any(T.issuperset(blocked_path_nodes) for blocked_path_nodes in blocked_semi_path):
            cond_two = True
        else:
            cond_two = all([set(path).intersection(na_xy.union(T0)) for path in paths])
            if cond_two:
                # cache the T, because any superset will also satisfy condition two
                blocked_semi_path.add(T)

        # Since both conditions hold, we can evaluate the insert operation
        if cond_one and cond_two:
            # scores are a function of the parents of the target node
            pa_target = set(G.predecessors(target))
            old_score = score_cache.local_score(data, target, pa_target)
            new_score = score_cache.local_score(data, target, pa_target.union({origin}))
            delta_score = new_score - old_score
            valid_operators.append((origin, target, T))
            scores.append(delta_score)
    return scores, valid_operators


def _score_delete_operators(origin, target, data, context: Context, score_cache: ScoreFunction):
    """Scores a valid delete operator of ``origin -> target``.

    A valid delete operator is described in Theorem 17 of :footcite:`chickering2002optimal`.
    There are two conditions that must be satisfied:

    - $NA_{X, Y} \\backslash H$ forms a clique.

    Here, $NA_{X, Y}$ is the set of nodes that are both undirected neighbors to X and
    adjacent to Y.

    $H$ is the set of nodes that are undirected neighbors with target that are adjacent
    to origin.

    Parameters
    ----------
    origin : Node
        Origin node.
    target : Node
        Target node.
    data : pd.DataFrame
        The dataset.
    context : Context
        Causal context.
    score_cache : Callable
        The scoring function, which takes a dataset, the origin node and target node,
        and returns a score.

    Returns
    -------
    scores : list
        List of delta scores for each valid operator.
    valid_operators : list
        List of valid operators, where each operator is a tuple of the form
        ``(origin, target, T)``, where ``origin`` is the origin node, ``target``
        is the target node, and ``T`` is the set of nodes that are adjacent to target that
        will be oriented pointing towards target.
    """
    G = context.init_graph

    valid_operators = []
    scores = []

    # get the superset H, which is the undirected neighbors of target (Y) that are also
    # adjacent to origin (X)
    origin_target_nbrs = set(undir_nbrs(G, target)).intersection(G.neighbors(origin))

    # get the undirected nbrs of both origin and adjacent to target
    na_xy = undir_nbrs(G, origin).intersection(G.neighbors(target))

    # define a cache that fast-checks cliques and fast-checks
    # subset of T that block the semi-directed paths from origin to target
    clique_cache: Set = set()

    for H in powerset(origin_target_nbrs):
        H = frozenset(H)

        # any superset of a known clique is also a clique
        if any(H.issuperset(clique) for clique in clique_cache):
            cond_one = True
        else:
            # condition 1: na_xy union target_nbrs forms a clique
            cond_one = is_clique(G, na_xy.difference(H))

            if cond_one:
                # remove any subset of target_nbrs that contains T as a subset
                # since the clique condition will also not hold for those
                clique_cache.add(H)

        if cond_one:
            # scores are a function of the parents of the target node
            pa_target = set(G.predecessors(target))
            old_score = score_cache.local_score(data, target, pa_target)
            new_score = score_cache.local_score(data, target, pa_target.difference({origin}))
            delta_score = new_score - old_score

            valid_operators.append((origin, target, H))
            scores.append(delta_score)
    return scores, valid_operators


def _score_essential_turn_operators(
    origin, target, data, context: Context, score_cache: ScoreFunction
):
    """Scores a valid turn operator for ``origin`` and ``target``.

    In an essential graph, the edge to be turned is directed already.
    A turn operation would change turn ``origin -> target``
    into ``origin <- target``.

    A valid turn operator is described in Proposition 34 of :footcite:`hauser2012characterization`.
    There are two conditions that must be satisfied for an edge $X \\rightarrow Y$:

    - $C \subset C0 = Nbr_{X,Y} \cup Adj_{Y}$ forms a clique.
    - $N = Nbr_{X} \cap Adj_{Y}$ is a subset of C
    - every path from X to Y in G except (X, Y) is blocked by $C \cup Nbr_{Y}$

    Here, $Nbr_{X}$ is the set of nodes that are undirected neighbors to X. C is a strict subset
    of $Nbr_{X}$. $Adj_{Y}$ is the set of nodes that are adjacent to Y. Neighbors
    imply any nodes that have an undirected edge with respect to the node.
    Adjacencies are any nodes that have any edge edge with respect to the node.

    Parameters
    ----------
    origin : Node
        Origin node.
    target : Node
        Target node.
    data : pd.DataFrame
        The dataset.
    context : Context
        Causal context.
    score_cache : Callable
        The scoring function, which takes a dataset, the origin node and target node,
        and returns a score.

    Returns
    -------
    scores : list
        List of delta scores for each valid operator.
    valid_operators : list
        List of valid operators, where each operator is a tuple of the form
        ``(origin, target, T)``, where ``origin`` is the origin node, ``target``
        is the target node, and ``T`` is the set of nodes that are adjacent to target that
        will be oriented pointing towards target.

    Notes
    -----
    The new score is computed by scoring any node with new parents, which affects both the origin
    and target in the turn operation. In this setting, a score is

    .. math::

        NewScore = S_G(Y, Pa_Y \\cup C \\cup {X}) + S_G(X, Pa_X \\backslash {Y}) \\
        OldScore = S_G(Y, Pa_Y \\cup C) + S_G(X, Pa_X)

    The change in the score is simply then the new score minus the old score.
    """
    undirected_edge_name = "undirected"
    G = context.init_graph
    C0 = set(G.get_graphs(edge_type=undirected_edge_name).neighbors(origin)) - set(
        G.neighbors(target)
    )

    valid_operators = []
    scores = []

    nonclique_cache: Set = set()
    blocked_semi_paths: Set = set()

    # get the undirected neighbors of target
    undir_target_nbrs = set(G.get_graphs(edge_type=undirected_edge_name).neighbors(target)) - {
        origin
    }
    origin_target_nbrs = undir_target_nbrs.intersection(G.neighbors(origin))
    target_nbrs = undir_nbrs(G, target)

    # get all semi-directed paths from origin to target
    paths = list(all_semi_directed_paths(G, origin, target))

    # now examine all possible subsets of undirected neighbors of target with
    # at least one node that is not adjacent to origin
    for T in powerset(C0):
        T = frozenset(T)

        # condition 1: C is a clique in subgraph of chain-component of target
        C = origin_target_nbrs | T
        if any(T.issuperset(nonclique) for nonclique in nonclique_cache):
            continue

        if not is_clique(G, C):
            nonclique_cache.add(T)
            continue

        # condition 2: all semi-directed paths from origin to target are blocked
        # by a member of C \union Nbr_{Y}
        # if any existing known blocked path is a superset of C, then it is for sure blocked
        if any(T.issuperset(blocked_path_nodes) for blocked_path_nodes in blocked_semi_paths):
            cond_two = True
        else:
            cond_two = all(
                [
                    set(path).intersection(T.union(target_nbrs))
                    for path in paths
                    if path != (origin, target)
                ]
            )
            if cond_two:
                # cache the T, because any superset will also satisfy condition two
                blocked_semi_paths.add(T)

        # condition 1 and condition 2 are met
        if cond_two:
            # scores are a function of the parents of the target node
            pa_target = set(G.predecessors(target))
            pa_origin = set(G.predecessors(origin))
            old_score = score_cache.local_score(
                data, target, pa_target.union(T)
            ) + score_cache.local_score(data, origin, pa_origin)
            new_score = score_cache.local_score(
                data, target, pa_target.union({origin}).union(T)
            ) + score_cache.local_score(data, origin, pa_origin.difference({target}))
            delta_score = new_score - old_score

            valid_operators.append((origin, target, T))
            scores.append(delta_score)
    return scores, valid_operators


def _score_nonessential_turn_operators(
    origin, target, data, context: Context, score_cache: ScoreFunction
):
    """Scores a valid turn operator for ``origin`` and ``target``.

    In a nonessential graph, the edge to be turned is not directed yet.
    A turn operation would change turn ``origin -- target``
    into ``origin <- target``.

    In :footcite:`hauser2012characterization`, Proposition 31, demonstrates
    that there exists no valid turn operator for an undirected edge if:

    1. $Nbr_{Y}$ are adjacent to X, or
    2. $Nbr_{Y} = {X}$

    A valid turn operator is described in Proposition 31 of :footcite:`hauser2012characterization`.
    There are two conditions that must be satisfied for an edge $X \\rightarrow Y$:

    1. $C = Nbr_{Y} - {X}$ forms a clique in the chain component of Y.
    2. all semi-directed paths from X to Y, except (X, Y) are blocked by $C \cup Nbr_{Y}$

    Here, $Nbr_{X}$ is the set of nodes that are undirected neighbors to X. C is a strict subset
    of $Nbr_{X}$. $Adj_{Y}$ is the set of nodes that are adjacent to Y. Neighbors
    imply any nodes that have an undirected edge with respect to the node.
    Adjacencies are any nodes that have any edge edge with respect to the node.

    Parameters
    ----------
    origin : Node
        Origin node.
    target : Node
        Target node.
    data : pd.DataFrame
        The dataset.
    context : Context
        Causal context.
    score_cache : Callable
        The scoring function, which takes a dataset, the origin node and target node,
        and returns a score.

    Returns
    -------
    scores : list
        List of delta scores for each valid operator.
    valid_operators : list
        List of valid operators, where each operator is a tuple of the form
        ``(origin, target, T)``, where ``origin`` is the origin node, ``target``
        is the target node, and ``T`` is the set of nodes that are adjacent to target that
        will be oriented pointing towards target.
    """
    undirected_edge_name = "undirected"

    G = context.init_graph

    valid_operators: List[Tuple] = []
    scores: List[float] = []

    nonclique_cache: Set = set()

    # get the undirected neighbors of target that is not the origin
    undir_target_nbrs = undir_nbrs(G, origin) - {origin}
    origin_target_nbrs = undir_target_nbrs.intersection(G.neighbors(origin))

    # get the undirected nbrs of both origin and adjacent to target
    na_xy = undir_nbrs(G, origin).intersection(G.neighbors(target))

    # check condition ii) of Proposition 31
    if len(undir_target_nbrs - origin_target_nbrs) == 0:
        return scores, valid_operators

    # get the chain component of the target
    target_chain_component = nx.node_connected_component(G.get_graphs(undirected_edge_name), target)
    target_ch_graph = G.get_graphs(undirected_edge_name).subgraph(target_chain_component)

    # now examine all possible subsets of undirected neighbors of target with
    # at least one node that is not adjacent to origin
    for T in powerset(undir_target_nbrs - origin_target_nbrs):
        T = frozenset(T)

        if any(T.issuperset(nonclique) for nonclique in nonclique_cache):
            continue

        # condition 1: C is a clique in subgraph of chain-component of target
        if not is_clique(G, origin_target_nbrs.union(T)):
            nonclique_cache.add(T)
            continue

        # condition 3: C \cap N and {origin, target} separates C and N \ C in
        # chain-component of target
        if not nx.d_separated(target_ch_graph, T, origin_target_nbrs - T, {origin, target}):
            continue

        # Since both conditions hold and condition ii) was validated at the beginning
        # of the function call, we can evaluate the insert operation
        # scores are a function of the parents of the target node
        pa_target = set(G.predecessors(target))
        pa_origin = set(G.predecessors(origin))

        old_score = score_cache.local_score(
            data, target, pa_target.union(T)
        ) + score_cache.local_score(
            data, origin, pa_origin.union(T.intersection(na_xy).union({target}))
        )
        new_score = score_cache.local_score(
            data, target, pa_target.union({origin}).union(T)
        ) + score_cache.local_score(data, origin, pa_origin.union(T.intersection(na_xy)))
        delta_score = new_score - old_score

        valid_operators.append((origin, target, T))
        scores.append(delta_score)
    return scores, valid_operators


def _forward_step(
    data: pd.DataFrame, context: Context, max_parents: int, score_cache: ScoreFunction
):
    """Score valid insert operations that can be applied to the current graph.

    Applies a greedy forward step to the graph, where each node is considered in turn
    and each possible parent is considered for that node. The parent that gives the
    best score is added to the graph.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    context : Context
        Causal context. The graph is stored in the context and is updated in-place.
    max_parents : int
        The maximum number of parents for each node.
    score_cache : ScoreFunction
        The scoring function, which takes in order the dataset, the origin node and a
        set of parent nodes, and returns a score.

    Returns
    -------
    delta_score : float
        The delta score of the best insert operation.
    G : Graph
        The updated graph.
    """
    directed_edge_name = "directed"

    # get a pointer to the graph
    G = context.init_graph

    # find all non-adjacent nodes
    nodes = set(G.nodes)

    # iterate over all nodes
    scores = []
    insert_ops = []
    for origin in nodes:
        for target in nodes:
            if origin == target:
                continue
            if G.has_edge(origin, target, edge_type=directed_edge_name):
                continue

            # now score all valid insert operations
            delta_scores, valid_insert_ops = _score_insert_operators(
                origin, target, data, context, score_cache
            )

            scores.extend(delta_scores)
            insert_ops.extend(valid_insert_ops)

    if len(insert_ops) == 0:
        return 0, G
    else:
        # take the best insert operation and apply it
        best_op_idx = np.argmax(scores)
        best_op = insert_ops[best_op_idx]

        # apply the best insert operation
        origin, target, T = best_op
        G.add_edge(origin, target, edge_type=directed_edge_name)
        for node in T:
            G.remove_edge(node, target)
            G.add_edge(node, target, edge_type=directed_edge_name)

    return scores[best_op_idx], G


def _backward_step(data, context, max_parents, score_cache: ScoreFunction):
    """Score valid delete operations that can be applied to the current graph.

    Applies a greedy backward step to the graph, where each node is considered in turn
    and each parent is considered for that node to be deleted. The parent-edge that gives the
    best score is removed from the graph.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    context : Context
        Causal context. The graph is stored in the context and is updated in-place.
    max_parents : int
        The maximum number of parents for each node.
    score_cache : ScoreFunction
        The scoring function, which takes in order the dataset, the origin node and
        a set of parent nodes, and returns a score.

    Returns
    -------
    delta_score : float
        The delta score of the best insert operation.
    G : Graph
        The updated graph.
    """
    directed_edge_name = "directed"

    # get a pointer to the graph
    G = context.init_graph

    # find all non-adjacent nodes
    nodes = set(G.nodes)

    # iterate over all nodes
    scores = []
    delete_ops = []
    for target in nodes:
        for origin in G.predecessors(target):
            if origin == target:
                continue
            if not G.has_edge(origin, target, directed_edge_name):
                continue

            # now score all valid delete operations
            delta_scores, valid_insert_ops = _score_delete_operators(
                origin, target, data, context, score_cache
            )

            scores.extend(delta_scores)
            delete_ops.extend(valid_insert_ops)

    if len(delete_ops) == 0:
        return 0, G
    else:
        # take the best insert operation and apply it
        best_op_idx = np.argmax(scores)
        best_op = delete_ops[best_op_idx]

        # apply the best delete operation
        origin, target, H = best_op
        G.remove_edge(origin, target, edge_type=directed_edge_name)
        for node in H:
            G.remove_edge(node, target)
            G.add_edge(node, target, edge_type=directed_edge_name)

    return scores[best_op_idx], G


def _turning_step(data, context, max_parents, score_cache: ScoreFunction):
    directed_edge_name = "directed"

    # get a pointer to the graph
    G = context.init_graph

    # find all non-adjacent nodes
    nodes = set(G.nodes)

    # iterate over all nodes
    scores = []
    delete_ops = []
    for target in nodes:
        for origin in G.neighbors(target):
            if origin == target:
                continue
            if not G.has_edge(origin, target):
                continue

            # now score all valid turn operations
            if G.has_edge(origin, target, directed_edge_name):
                delta_scores, valid_insert_ops = _score_essential_turn_operators(
                    origin, target, data, context, score_cache
                )
            else:
                delta_scores, valid_insert_ops = _score_nonessential_turn_operators(
                    origin, target, data, context, score_cache
                )

            scores.extend(delta_scores)
            delete_ops.extend(valid_insert_ops)

    if len(delete_ops) == 0:
        return 0, G
    else:
        # take the best insert operation and apply it
        best_op_idx = np.argmax(scores)
        best_op = delete_ops[best_op_idx]

        # apply the best turn operation
        origin, target, C = best_op
        G = _apply_turn(G, origin, target, C)

    return scores[best_op_idx], G


def _apply_turn(G, origin, target, C):
    """Apply turn operator on the graph.

    For an edge ``origin -> target``, or ``origin -- target``, the turn operator
    changes the edge to ``origin <- target``. Then for all nodes, ``c`` in ``C``, we
    orient the previously undirected edge ``c -> origin``.

    Parameters
    ----------
    G : Graph
        The mixed-edge graph.
    origin : Node
        The origin node.
    target : Node
        The target node.
    C : set
        The set of nodes.

    Returns
    -------
    G : Graph
        The updated graph.
    """
    # turn edge origin -> target into origin <- target
    G.remove_edge(origin, target)
    G.add_edge(target, origin, edge_type="directed")

    # for all nodes c in C, orient the previously undirected edge c -> origin
    for node in C:
        G.remove_edge(node, origin)
        G.add_edge(node, origin, edge_type="directed")
    return G


class GreedyEquivalentSearch:
    def __init__(
        self,
        forward_phase: bool = True,
        backward_phase: bool = True,
        turning_phase: bool = True,
        max_parents: Optional[int] = None,
        metric="bic",
        iterate: bool = True,
    ) -> None:
        """Greedy equivalent search for causal Bayesian network structure learning.

        Implements the GES algorithm initially introduced in :footcite:`chickering2002optimal`
        for any decomposable score function.

        Parameters
        ----------
        forward_phase : bool, optional
            Whether to run the forward-phase of the algorithm, by default True.
            See Theorem 15 of :footcite:`chickering2002optimal`.
        backward_phase : bool, optional
            Whether to run the backward-phase of the algorithm, by default True.
            See Theorem 17 of :footcite:`chickering2002optimal`.
        turning_phase : bool, optional
            Whether to run the turning-phase of the algorithm, by default True.
            See Theorem of :footcite:`hauser2012characterization`.
        max_parents : Optional[int], optional
            The maximum number of parents for each node, by default None, which implies
            no limit on the number of parents.
        metric : str, optional
            The score function, by default 'bic'. The score function is assumed to have the
            "score equivalence" and "decomposability" properties.
        iterate : bool, optional
            Whether to iterate until convergence, by default True.

        Notes
        -----
        For details on the insert and delete operators, see Theorem 15 and Theorem 17 of
        :footcite:`chickering2002optimal`.

        The turning step was introduced in :footcite:`hauser2012characterization` in
        Proposition 31 and Proposition 34.

        Other improvements were made in improving the runtime efficiency of the search algorithm,
        such as :footcite:`chickering2015selective` and :footcite:`chickering2020statistically`.

        Score functions are typically some form of likelihood function with some penalty term
        on the complexity of the graph structure. The BIC score is the most common, but other
        score functions are possible, such as the AIC score. If the score function has the property
        of "score equivalence", then it forms an equivalence class of graphs that have the same
        score for the same data. If the score function has the property of "decomposability", then
        it can be decomposed into a sum of scores for each node, which can be computed independently
        and in parallel, which is a desirable computational property.

        References
        ----------
        .. footbibliography::
        """
        self.forward_phase = forward_phase
        self.backward_phase = backward_phase
        self.turning_phase = turning_phase
        self.max_parents = max_parents
        self.metric = metric

        self.iterate = iterate

    def learn_graph(self, data: pd.DataFrame, context: Context):
        if all(
            not phase for phase in [self.forward_phase, self.backward_phase, self.turning_phase]
        ):
            raise ValueError("At least one phase must be True.")

        phase_funcs: List[Callable] = []
        for phase, phase_name in zip(
            [self.forward_phase, self.backward_phase, self.turning_phase],
            [_forward_step, _backward_step, _turning_step],
        ):
            if phase:
                phase_funcs.append(phase_name)

        # define the cache
        score_cache = ScoreFunction(self.metric)

        # score the initial graph
        # total_score = score_cache.full_score(data, context.init_graph)
        total_score = 0.0
        last_total_score = -np.inf
        delta_score = np.inf

        while total_score > last_total_score and self.iterate:
            last_total_score = total_score

            # loop through each phase and iterate until the score does not change
            for phase_func in phase_funcs:
                while delta_score > THRESHOLD:
                    delta_score, new_G = phase_func(data, context, self.max_parents, score_cache)

                    # if the score changes, then update the graph and the total score
                    if delta_score > 1e-6:
                        G = pdag_to_cpdag(new_G)
                        context.init_graph = G
                        total_score += delta_score
                    else:
                        break

        G = context.init_graph
        self.context_ = context.copy()
        self.graph_ = G

        return self
