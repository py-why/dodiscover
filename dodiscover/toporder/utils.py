from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def full_dag(top_order: List[int]) -> NDArray:
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


def orders_consistency(order_full: List[int], order_noleaf: List[int]) -> bool:
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


def dummy_sample(G: nx.DiGraph = None, seed: int = 42, n_samples=100) -> pd.DataFrame:
    """Generate data from an additive noise model.

    Data are generated from a Structural Causal Model consistent with the input graph G.
    Nonlinear functional mechanisms are a simple sum of the sine of the parents node.

    Parameters
    ----------
    G : nx.DiGraph, optional
        Directed acyclic graph. If None (default) get the groundtruth from `dummy_groundtruth()`
        method of dodiscover.toporder.utils module.
    seed : int, optional
        Fixed random seed, default is 42.
    n_samples : int, optional
        Number of samples in the dataset, default is 100.

    Return
    ------
    pd.DataFrame
        Pandas dataframe of samples generated according to the input DAG from an
        additive noise model.
    """
    if G is None:
        G = dummy_groundtruth()
    assert nx.is_directed_acyclic_graph(G), "Input graph must be a DAG"
    np.random.seed(seed)
    A = nx.to_numpy_array(G)
    order = list(nx.topological_sort(G))
    X = np.random.randn(n_samples, len(order)) * np.random.uniform(0.5, 1)  # sample noise
    for node in order:  # iterate starting from source
        parents = np.flatnonzero(A[:, node])
        if len(parents) > 0:
            X[:, node] += np.sum(np.sin(X[:, parents]), axis=1)
    return pd.DataFrame(X)


def dummy_groundtruth() -> None:
    """
    Ground truth associated to dummy_sample dataset
    """
    A = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0]])
    return nx.from_numpy_array(A, create_using=nx.DiGraph)


def dummy_dense() -> None:
    """
    Dense adjacency matrix associated to order = [2, 1, 3, 0]
    """
    A = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0]])
    return nx.from_numpy_array(A, create_using=nx.DiGraph)
