from typing import Set

import networkx as nx
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
    nbrs = set()

    # query all neighbors of X and then only add nodes that are in a valid path
    # to end
    for node in G.neighbors(start):
        if not G.has_edge(start, node):
            raise RuntimeError(f"{start} and {node} are not connected, but they are assumed to be.")

        # if we queried the edge we are testing, then pick that one
        if node == end:
            continue

        # find a path from start node to end
        paths = nx.all_simple_paths(G, source=node, target=end)
        for path in paths:
            # the trivial path which indicates that 'node' is only connected to
            # 'end' through 'start'
            if path == (node, start, end):
                continue
            else:
                # found a single path
                nbrs.add(node)
                break
    return nbrs
