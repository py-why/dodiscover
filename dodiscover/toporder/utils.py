from typing import List

import numpy as np
from numpy.typing import NDArray


def full_DAG(top_order: List[int]) -> NDArray:
    """Find adjacency matrix of the fully connected DAG from the topological order.

    Parameters
    ----------
    top_order : List[int]
        Topological order of nodes in a causal graph.

    Return
    ------
    A : np.ndarray
        The fully connected adjacency matrix encoding the input topological order.
    """
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i + 1 :]] = 1
    return A


def full_adj_to_order(A: NDArray) -> List[int]:
    """Find topological ordering from the adjacency matrix A.

    Parameters
    ----------
    A : np.ndarray
        The fully connected adjacency matrix encoding the input topological order.

    Return
    ------
    top_order : List[int]
        Topological order encoding of the input fully connected adjacency matrix.
    """
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order


def orders_consistency(order_full, order_noleaf) -> bool:
    """Check consistency of topological order with and without a single leaf.

    Parameters
    ----------
    order_full : List[int]
        Inferred topological order on the full graph.
    order_noleaf : List[int]
        Inferred topological order on the graph pruned by a leaf.

    Return
    ------
    bool
        True if the two orders are consistent    
    """
    for node in order_noleaf:
        if node not in order_full:
            return False
    return True
