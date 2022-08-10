import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from .constraint._protocol import GraphProtocol


def graph_to_pred_vector(graph, sort_first: bool = False, as_adjacency: bool = True):
    """Convert a causal DAG graph to a prediction vector.

    If you use two graphs, you first have to align the nodes.

    Parameters
    ----------
    graph : instance of causal DAG
        The causal graph.
    sort_first : bool
        Whether to sort the graph's nodes ordering and order the adjacency matrix
        representation that way. By default False.
    as_adjacency : bool, optional
        Whether to convert to a vector representing the adjacencies, by default True.
        If False, then will also evaluate correct edges. See Notes for details.

    Returns
    -------
    y_pred : np.ndarray of shape ((n_nodes ** 2 - n_nodes) / 2,)
        _description_

    Notes
    -----
    Assumes that the graph is acyclic.
    """
    # convert graphs to adjacency graph in networkx
    if not isinstance(graph, np.ndarray):
        graph = graph.to_undirected()  # type: ignore

    # next convert into 2D numpy array format
    adj_mat = nx.to_numpy_array(graph)

    if sort_first:
        # get the order of the nodes
        idx = np.argsort(graph.nodes)

        # next convert into 2D numpy array format and make sure nodes are ordered accordingly
        adj_mat = adj_mat[np.ix_(idx, idx)]

    # finding advanced indices of upper/lower right triangle
    triu_idx = np.triu_indices_from(adj_mat, k=1)
    tril_idx = np.tril_indices_from(adj_mat, k=-1)

    if as_adjacency:
        # then only extract lower-triangular portion and binarize the labels
        adj_mat = adj_mat[tril_idx]
        adj_mat = adj_mat > 0
        y_vec = LabelBinarizer().fit_transform(adj_mat.flatten()).squeeze()
    else:
        raise RuntimeError("Doesn't work yet...")
        out = np.ones(adj_mat.shape, dtype=bool)
        out[tril_idx] = False
        out[triu_idx] = False
        adj_mat = adj_mat[out]
        y_vec = LabelEncoder().fit_transform(adj_mat.flatten()).squeeze()
    return y_vec


def confusion_matrix_networks(
    true_graph,
    pred_graph,
):
    """Compute the confusion matrix comparing a predicted graph from the true graph.

    Converts the graphs into adjacency matrices, which are symmetric.

    Parameters
    ----------
    true_graph : an instance of causal graph
        The true graph.
    pred_graph : an instance of causal graph
        The predicted graph. The predicted graph and true graph must be
        the same type.

    Returns
    -------
    cm : np.ndarray of shape (2, 2)
        The confusion matrix.

    See Also
    --------
    sklearn.metrics.confusion_matrix
    """
    assert set(true_graph.nodes) == set(pred_graph.nodes)

    # convert graphs to adjacency graph in networkx
    if isinstance(true_graph, GraphProtocol):
        true_graph = true_graph.to_undirected()
    if isinstance(pred_graph, GraphProtocol):
        pred_graph = pred_graph.to_undirected()

    # get the order of the nodes
    idx = np.argsort(true_graph.nodes)
    other_idx = np.argsort(pred_graph.nodes)

    # next convert into 2D numpy array format and make sure nodes are ordered accordingly
    true_adj_mat = nx.to_numpy_array(true_graph)[np.ix_(idx, idx)]
    pred_adj_mat = nx.to_numpy_array(pred_graph)[np.ix_(other_idx, other_idx)]

    # ensure we are looking at symmetric graphs
    true_adj_mat += true_adj_mat.T
    pred_adj_mat += pred_adj_mat.T

    # then only extract lower-triangular portion
    true_adj_mat = true_adj_mat[np.tril_indices_from(true_adj_mat, k=-1)]
    pred_adj_mat = pred_adj_mat[np.tril_indices_from(pred_adj_mat, k=-1)]

    true_adj_mat = true_adj_mat > 0
    pred_adj_mat = pred_adj_mat > 0

    # vectorize and binarize for sklearn's confusion matrix
    y_true = LabelBinarizer().fit_transform(true_adj_mat.flatten()).squeeze()
    y_pred = LabelBinarizer().fit_transform(pred_adj_mat.flatten()).squeeze()

    # compute the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return conf_mat


def structure_hamming_dist(graph, other_graph, double_for_anticausal: bool = True):
    """Compute structural hamming distance.

    Parameters
    ----------
    graph : _type_
        Reference graph.
    other_graph : _type_
        Other graph.
    double_for_anticausal : bool, optional
        Whether to count incorrect orientations as two mistakes, by default True

    Returns
    -------
    shm : float
        The hamming distance between 0 and infinity.
    """
    if isinstance(graph, GraphProtocol):
        graph = graph.to_networkx()  # type: ignore
    if isinstance(other_graph, GraphProtocol):
        other_graph = other_graph.to_networkx()  # type: ignore

    # get the order of the nodes
    idx = np.argsort(graph.nodes)
    other_idx = np.argsort(other_graph.nodes)

    # convert graphs to adjacency matrix in numpy array format
    adj_mat = nx.to_numpy_array(graph)[np.ix_(idx, idx)]
    other_adj_mat = nx.to_numpy_array(other_graph)[np.ix_(other_idx, other_idx)]

    diff = np.abs(adj_mat - other_adj_mat)

    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2
