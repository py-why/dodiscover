from typing import List, Optional

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from ._protocol import Graph
from .typing import NetworkxGraph


def confusion_matrix_networks(
    true_graph: Graph,
    pred_graph: Graph,
    labels: Optional[NDArray] = None,
    normalize: Optional[str] = None,
):
    """Compute the confusion matrix comparing a predicted graph from the true graph.

    Converts the graphs into an undirected graph, and then compares their adjacency
    matrix, which are symmetric.

    Parameters
    ----------
    true_graph : instance of causal graph
        The true graph.
    pred_graph : instance of causal graph
        The predicted graph. The predicted graph and true graph must be
        the same type.
    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    cm : np.ndarray of shape (2, 2)
        The confusion matrix.

    See Also
    --------
    sklearn.metrics.confusion_matrix

    Notes
    -----
    This function only compares the graph's adjacency structure, which does
    not take into consideration the directionality of edges.
    """
    if set(true_graph.nodes) != set(pred_graph.nodes):
        raise RuntimeError("Both nodes should match.")

    # convert graphs to undirected graph in networkx
    true_graph = true_graph.to_undirected()
    pred_graph = pred_graph.to_undirected()

    # get the order of the nodes
    idx = np.argsort(true_graph.nodes)
    other_idx = np.argsort(pred_graph.nodes)

    # next convert into 2D numpy array format and make sure nodes are ordered accordingly
    true_adj_mat = nx.to_numpy_array(true_graph)[np.ix_(idx, idx)]
    pred_adj_mat = nx.to_numpy_array(pred_graph)[np.ix_(other_idx, other_idx)]

    # then only extract lower-triangular portion
    true_adj_mat = true_adj_mat[np.tril_indices_from(true_adj_mat, k=-1)]
    pred_adj_mat = pred_adj_mat[np.tril_indices_from(pred_adj_mat, k=-1)]

    true_adj_mat = true_adj_mat > 0
    pred_adj_mat = pred_adj_mat > 0

    # vectorize and binarize for sklearn's confusion matrix
    y_true = LabelBinarizer().fit_transform(true_adj_mat.flatten()).squeeze()
    y_pred = LabelBinarizer().fit_transform(pred_adj_mat.flatten()).squeeze()

    # compute the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return conf_mat


def structure_hamming_dist(
    true_graph: NetworkxGraph, pred_graph: NetworkxGraph, double_for_anticausal: bool = True
) -> float:
    """Compute structural hamming distance.

    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing; the
    ``double_for_anticausal`` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.

    Parameters
    ----------
    true_graph : instance of nx.Graph or nx.DiGraph
        The true graph as an instance of a MixedEdgeGraph with only one type of
        edge.
    pred_graph : instance of nx.Graph or nx.DiGraph
        The predicted graph. The predicted graph and true graph must be
        the same type.
    double_for_anticausal : bool, optional
        Whether to count incorrect orientations as two mistakes, by default True

    Returns
    -------
    shd : float
        The hamming distance between 0 and infinity.

    Notes
    -----
    SHD is only well defined if you have a graph with only undirected edges,
    or directed edges. That is, we only consider a Bayesian network, or a causal
    DAG as candidates. If there are more than one type of edge within
    the network, then SHD can be called on a sub-graph of that edge type. For example,
    say you would like to compare a PAG, where there are directed, undirected, bidirected
    and edges with circular endpoints. Currently, there is no known way of comparing
    two PAGs systematically. Therefore, one can compare PAGs via the number of circle
    edges, or the SHD of the undirected, bidirected, directed edge subgraphs.
    """
    if type(true_graph) != type(pred_graph):
        raise RuntimeError(
            f"The type of graphs must be the same: {type(true_graph), type(pred_graph)}"
        )

    # get the order of the nodes
    idx = np.argsort(true_graph.nodes)
    other_idx = np.argsort(pred_graph.nodes)

    # convert graphs to adjacency matrix in numpy array format
    adj_mat = nx.to_numpy_array(true_graph)[np.ix_(idx, idx)]
    other_adj_mat = nx.to_numpy_array(pred_graph)[np.ix_(other_idx, other_idx)]

    diff = np.abs(adj_mat - other_adj_mat)

    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.T
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2


def toporder_divergence(true_graph: NetworkxGraph, order: List[int]) -> int:
    """Compute topological ordering divergence.

    Topological order divergence is used to compute the number of false negatives,
    i.e. missing edges, associated to a topological order of the nodes of a
    graph with respect to the ground truth structure.
    If the topological ordering is compatible with the graph ground truth,
    the divergence is equal to 0. In the worst case of completely reversed
    ordering, toporder_divergence is equals to P, the number of edges (positives)
    in the ground truth graph.
    Note that the divergence defines a lower bound for the Structural Hamming Distance.

    Parameters
    ----------
    true_graph : NetworkxGraph
        Input groundtruth directed acyclic graph.
    order : List[int]
        A topological ordering on the nodes of the graph.

    Returns
    -------
    err : int
        Sum of the number of edges of A not admitted by the given order.
    """
    if not nx.is_directed_acyclic_graph(true_graph):
        raise ValueError("The input graph must be directed and acyclic.")

    # convert graphs to adjacency matrix in numpy array format
    A = nx.to_numpy_array(true_graph)

    if len(order) != A.shape[0] or A.shape[0] != A.shape[1]:
        raise ValueError("The dimensions of the graph and the order list do not match.")

    false_negatives_from_order = 0
    for i in range(len(order)):
        false_negatives_from_order += A[order[i + 1 :], order[i]].sum()
    return false_negatives_from_order
