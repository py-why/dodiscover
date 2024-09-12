from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


# TODO: replace with pywhy-stats implementation
def kernel_width(X: NDArray):
    """
    Estimate width of the Gaussian kernel.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_nodes)
        Matrix of the data.
    """
    X_diff = np.expand_dims(X, axis=1) - X  # Gram matrix of the data
    D = np.linalg.norm(X_diff, axis=2).flatten()
    D_nonzeros = D[D > 0]  # Remove zeros
    s = np.median(D_nonzeros) if np.any(D_nonzeros) else 1
    return s


def full_dag(top_order: List[int]) -> NDArray:
    """Find adjacency matrix of the fully connected DAG from the topological order.

    Parameters
    ----------
    top_order : List[int]
        Topological order of nodes in a causal graph.

    Returns
    -------
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

    Returns
    -------
    top_order : List[int]
        Topological order encoding of the input fully connected adjacency matrix.
    """
    order = list(A.sum(axis=1).argsort())
    order.reverse()  # reverse to get order starting with source nodes
    return order


def orders_consistency(order_full: List[int], order_noleaf: List[int]) -> bool:
    """Check consistency of topological order with and without a single leaf.

    Parameters
    ----------
    order_full : List[int]
        Inferred topological order on the full graph.
    order_noleaf : List[int]
        Inferred topological order on the graph pruned by a leaf.

    Returns
    -------
    bool
        True if the two orders are consistent
    """
    for node in order_noleaf:
        if node not in order_full:
            return False
    return True


def dummy_sample(G: Optional[nx.DiGraph] = None, seed: int = 42, n_samples=100) -> pd.DataFrame:
    """Generate data from an additive noise model.

    Data are generated from a Structural Causal Model consistent with the input graph G.
    Nonlinear functional mechanisms are a simple sum of the sine of the parents node.

    Parameters
    ----------
    G : nx.DiGraph, optional
        Directed acyclic graph. If None (default) get the groundtruth from `dummy_groundtruth`
        method of dodiscover.toporder.utils module.
    seed : int, optional
        Fixed random seed, default is 42.
    n_samples : int, optional
        Number of samples in the dataset, default is 100.

    Returns
    -------
    data : pd.DataFrame
        Pandas dataframe of samples generated according to the input DAG from an
        additive noise model.
    """
    if G is None:
        G = dummy_groundtruth()
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Input graph must be a DAG")
    np.random.seed(seed)
    A = nx.to_numpy_array(G)
    order = list(nx.topological_sort(G))
    X = np.random.randn(n_samples, len(order)) * np.random.uniform(0.5, 1)  # sample noise
    for node in order:  # iterate starting from source
        parents = np.flatnonzero(A[:, node])
        if len(parents) > 0:
            X[:, node] += np.sum(np.sin(X[:, parents]), axis=1)
    data = pd.DataFrame(X)
    return data


def dummy_groundtruth() -> nx.DiGraph:
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


# Preliminary Neighbours Search
def pns(A: NDArray, X: NDArray, pns_threshold, pns_num_neighbors) -> NDArray:
    """Preliminary Neighbors Selection (PNS) pruning on adjacency matrix `A`.

    Variable selection preliminary to CAM pruning.
    PNS :footcite:`Buhlmann2013` allows to scale CAM pruning to large graphs
    (~20 or more nodes), with sensitive reduction of computational time.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix representation of a dense graph.
    X : np.ndarray of shape (n_samples, n_nodes)
        Dataset with observations of the causal variables.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    Returns
    -------
    A : np.ndarray
        Pruned adjacency matrix.

    References
    ----------
    .. footbibliography::
    """
    num_nodes = X.shape[1]
    for node in range(num_nodes):
        X_copy = np.copy(X)
        X_copy[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(X_copy, X[:, node])
        selected_reg = SelectFromModel(
            reg,
            threshold="{}*mean".format(pns_threshold),
            prefit=True,
            max_features=pns_num_neighbors,
        )
        mask_selected = selected_reg.get_support(indices=False)

        mask_selected = mask_selected.astype(A.dtype)
        A[:, node] *= mask_selected

    return A
