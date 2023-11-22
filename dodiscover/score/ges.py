from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from pywhy_graphs.algorithms import all_semi_directed_paths

from dodiscover.context import Context

from .score_function import ScoreFunction
from .utils import is_clique, powerset


def _score_insert_operators(origin, target, data, context: Context, score_cache: ScoreFunction):
    """Scores a valid insert operation of ``origin -> target``.

    A valid insert operator is described in Theorem 15 of :footcite:`chickering2002optimal`.
    There are two conditions that must be satisfied:

    - $NA_{X, Y} \cup T$ forms a clique.
    - Every semi-directed path from origin to target is blocked by a node from $NA_{X, Y} \cup T$.

    Here, $NA_{X, Y}$ is the set of nodes that are adjacent to both origin and target. $T$ is
    the set of nodes that are adjacent to target (i.e. target nbrs).

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

    # get the target node's nbrs
    target_nbrs = set(G.neighbors(target))

    # get the nbrs of both origin and target
    na_yx = set(G.neighbors(origin)).intersection(set(G.neighbors(target)))

    # get all semi-directed paths from origin to target
    paths = list(all_semi_directed_paths(G, origin, target))

    # define a cache that fast-checks cliques and fast-checks
    # subset of T that block the semi-directed paths from origin to target
    nonclique_cache = set()
    blocked_semi_path = set()

    for T in powerset(target_nbrs):
        T = frozenset(T)

        # any superset of a known nonclique is also a nonclique
        if any(T.issuperset(nonclique) for nonclique in nonclique_cache):
            continue

        # condition 1: na_yx union target_nbrs forms a clique
        cond_one = is_clique(G, na_yx.union(T))
        if not cond_one:
            # remove any subset of target_nbrs that contains T as a subset
            # since the clique condition will also not hold for those
            nonclique_cache.add(T)

        # condition 2: every semi-directed path from origin to target is
        # blocked by a node from na_yx union target_nbrs

        # if any existing known blocked path is a superset of T, then it is for sure blocked
        if any(T.issuperset(blocked_path_nodes) for blocked_path_nodes in blocked_semi_path):
            cond_two = True
        else:
            cond_two = all([set(path).intersection(na_yx.union(target_nbrs)) for path in paths])
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

    Here, $NA_{X, Y}$ is the set of nodes that are adjacent to both origin and target. $H$ is
    the set of nodes that are adjacent to both target and origin nodes.

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

    # get the target node's nbrs
    origin_target_nbrs = set(G.neighbors(target)).union(G.neighbors(origin))

    # get the nbrs of both origin and target
    na_yx = set(G.neighbors(origin)).intersection(set(G.neighbors(target)))

    # define a cache that fast-checks cliques and fast-checks
    # subset of T that block the semi-directed paths from origin to target
    clique_cache = set()

    for H in powerset(origin_target_nbrs):
        H = frozenset(H)

        # any superset of a known clique is also a clique
        if any(H.issuperset(clique) for clique in clique_cache):
            cond_one = True
        else:
            # condition 1: na_yx union target_nbrs forms a clique
            cond_one = is_clique(G, na_yx.difference(H))

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

    - $C = NA_{X,Y} \cup T$ forms a clique.
    - $N = NA_{X} \cap Adj_{Y}$ is a subset of C
    - every path from X to Y in G except (X, Y) is blocked by $C \cup Na_{Y}$

    Here, $NA_{X}$ is the set of nodes that are neighbors to X. C is a strict subset
    of $NA_{X}$. $Adj_{Y}$ is the set of nodes that are adjacent to Y. Neighbors
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
    pass


def _score_nonessential_turn_operators(
    origin, target, data, context: Context, score_cache: ScoreFunction
):
    """Scores a valid turn operator for ``origin`` and ``target``.

    In a nonessential graph, the edge to be turned is not directed yet.
    A turn operation would change turn ``origin -- target``
    into ``origin <- target``.

    In :footcite:`hauser2012characterization`, Proposition 31, demonstrates
    that there exists no valid turn operator for an undirected edge if:

    1. $NA_{Y}$ are adjacent to X, or
    2. $NA_{Y} = {X}$

    A valid turn operator is described in Proposition 34 of :footcite:`hauser2012characterization`.
    There are two conditions that must be satisfied for an edge $X \\rightarrow Y$:

    - $C = NA_{X,Y} \cup T$ forms a clique.
    - all semi-directed paths from X to Y, except (X, Y) are blocked by $C \cup NA_{Y}$

    Here, $NA_{X}$ is the set of nodes that are neighbors to X. C is a strict subset
    of $NA_{X}$. $Adj_{Y}$ is the set of nodes that are adjacent to Y. Neighbors
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

    valid_operators = []
    scores = []

    nonclique_cache = set()

    # get the undirected neighbors of target
    undir_target_nbrs = set(G.get_graphs(edge_type=undirected_edge_name).neighbors(target)) - {
        origin
    }
    origin_target_nbrs = undir_target_nbrs.intersection(G.neighbors(origin))

    # check condition ii) of Proposition 31
    if len(undir_target_nbrs - origin_target_nbrs) == 0:
        return scores, valid_operators

    # get the chain component of the target
    target_chain_component = nx.node_connected_component(G.get_graphs(undirected_edge_name), target)

    # now examine all possible subsets of undirected neighbors of target with
    # at least one node that is not adjacent to origin
    for T in powerset(undir_target_nbrs - origin_target_nbrs):
        T = frozenset(T)

        if any(T.issuperset(nonclique) for nonclique in nonclique_cache):
            continue

        # condition 1: C is a clique in subgraph of chain-component of target
        cond_one = is_clique(G, origin_target_nbrs.union(T))
        if not cond_one:
            nonclique_cache.add(T)
            continue

        # condition 3: C \cap N and {origin, target} separates C and N \ C in chain-component of target
        cond_three = True

        # Since both conditions hold and condition ii) was validated at the beginning
        # of the function call, we can evaluate the insert operation
        if cond_one and cond_three:
            # scores are a function of the parents of the target node
            pa_target = set(G.predecessors(target))
            old_score = score_cache.local_score(data, target, pa_target)
            new_score = score_cache.local_score(data, target, pa_target.union({origin}))
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
        The scoring function, which takes in order the dataset, the origin node and a set of parent nodes,
        and returns a score.

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
        The scoring function, which takes in order the dataset, the origin node and a set of parent nodes,
        and returns a score.

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

        # TODO: fix
        # apply the best turn operation
        origin, target, H = best_op
        G.remove_edge(origin, target, edge_type=directed_edge_name)
        for node in H:
            G.remove_edge(node, target)
            G.add_edge(node, target, edge_type=directed_edge_name)

    return scores[best_op_idx], G


class GreedyEquivalentSearch:
    def __init__(
        self,
        forward_phase: bool = True,
        backward_phase: bool = True,
        turning_phase: bool = True,
        max_parents: Optional[int] = None,
        metric="bic",
    ) -> None:
        """Greedy equivalent search with an arbitrary score metric function for causal Bayesian network structure learning.

        Implements the GES algorithm initially introduced in :footcite:`chickering2002optimal`.

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

        Notes
        -----
        The turning step was introduced in :footcite:`hauser2012characterization`.

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

    def learn_graph(self, data: pd.DataFrame, context: Context):
        # define the cache
        score_cache = ScoreFunction(self.metric)

        if self.forward_phase:
            delta_score, G = _forward_step(data, context, self.max_parents, score_cache)
            while delta_score > 0:
                delta_score, G = _forward_step(data, context, self.max_parents, score_cache)
        if self.backward_phase:
            delta_score, G = _backward_step(data, context, self.max_parents, score_cache)
            while delta_score > 0:
                delta_score, G = _backward_step(data, context, self.max_parents, score_cache)
        if self.turning_phase:
            delta_score, G = _turning_step(data, context, self.max_parents, score_cache)
            while delta_score > 0:
                delta_score, G = _turning_step(data, context, self.max_parents, score_cache)

        G = context.init_graph
        self.context_ = context.copy()
        self.graph_ = G

        return self
