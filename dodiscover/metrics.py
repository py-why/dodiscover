from typing import Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from ._protocol import GraphProtocol


def confusion_matrix_networks(
    true_graph: GraphProtocol,
    pred_graph: GraphProtocol,
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
    if set(true_graph.nodes) != set(pred_graph.nodes):
        raise RuntimeError("Both nodes should match.")

    # convert graphs to adjacency graph in networkx
    true_graph = true_graph.to_undirected()
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
