from itertools import combinations
from typing import Iterable, Set, SupportsFloat, Union

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover import Graph
from dodiscover.typing import SeparatingSet


def dummy_sample(G: Graph):
    """Sample an empty dataframe with columns as the nodes.

    Used for oracle testing.
    """
    return pd.DataFrame({column: [] for column in G.nodes})  # type: ignore


def is_in_sep_set(
    check_var,
    sep_set: SeparatingSet,
    x_var,
    y_var,
    mode: str = "any",
):
    """Check that a variable is not in any separating set between 'X' and 'Y'.

    Parameters
    ----------
    check_var : node
        The variable to check.
    sep_set : Dict[str, Dict[str, List[Set[Any]]]]
        The separating sets between any two variables 'X' and 'Y'.
    x_var : node
        The 'X' variable.
    y_var : node
        The 'Y' variable.
    mode : str
        Whether to check in 'any' separating sets, or check if it is in 'all' separating sets,
        or if it is in 'some' separating sets, but not all. Default is 'any'.

    Returns
    -------
    bool
        Whether or not 'check_var' is in all separating sets of 'x_var' and 'y_var'.
    """
    if mode == "any":
        func = any
    elif mode == "all":
        func = all
    elif mode == "some":
        return any(check_var in _sep_set for _sep_set in sep_set[x_var][y_var]) and not all(
            check_var in _sep_set for _sep_set in sep_set[x_var][y_var]
        )
    return func(check_var in _sep_set for _sep_set in sep_set[x_var][y_var])


def _iter_conditioning_set(
    possible_variables: Iterable,
    x_var: Union[SupportsFloat, str],
    y_var: Union[SupportsFloat, str],
    size_cond_set: int,
) -> Iterable[Set]:
    """Iterate function to generate the conditioning set.

    Parameters
    ----------
    possible_variables : iterable
        A set/list/dict of possible variables to consider for the conditioning set.
        This can be for example, the current adjacencies.
    x_var : node
        The node for the 'x' variable.
    y_var : node
        The node for the 'y' variable.
    size_cond_set : int
        The size of the conditioning set to consider. If there are
        less adjacent variables than this number, then all variables will be in the
        conditioning set.

    Yields
    ------
    Z : set
        The set of variables for the conditioning set.
    """
    exclusion_set = {x_var, y_var}

    all_adj_excl_current = [p for p in possible_variables if p not in exclusion_set]

    # loop through all possible combinations of the conditioning set size
    for cond in combinations(all_adj_excl_current, size_cond_set):
        cond_set = set(cond)
        yield cond_set


def _find_neighbors_along_path(G: nx.Graph, start, end) -> Set:
    """Find neighbors that are along a path from start to end.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    start : Node
        The starting node.
    end : Node
        The ending node.

    Returns
    -------
    nbrs : Set
        The set of neighbors that are also along a path towards
        the 'end' node.
    """

    def _assign_weight(u, v, edge_attr):
        if u == node or v == node:
            return np.inf
        else:
            return 1

    nbrs = set()
    for node in G.neighbors(start):
        if not G.has_edge(start, node):
            raise RuntimeError(f"{start} and {node} are not connected, but they are assumed to be.")

        # find a path from start node to end
        path = nx.shortest_path(G, source=node, target=end, weight=_assign_weight)
        if len(path) > 0:
            if start in path:
                raise RuntimeError("There is an error with the input. This is not possible.")
            nbrs.add(node)
    return nbrs
