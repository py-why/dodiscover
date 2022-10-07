import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dodiscover.metrics import confusion_matrix_networks, structure_hamming_dist


def test_confusion_matrix_networks():
    seed = 12345
    G_true = nx.gnp_random_graph(5, 0.8, seed=seed)
    G_learned = nx.gnp_random_graph(5, 0.8, seed=seed)
    true_edges = set(G_true.edges)
    detected_edges = set(G_learned.edges)

    all_pairs = set(
        [(a, b) for idx, a in enumerate(list(G_true.nodes)) for b in list(G_true.nodes)[idx + 1 :]]
    )

    false_edges = all_pairs - true_edges

    TP = true_edges.intersection(detected_edges)
    FP = detected_edges - true_edges
    TN = false_edges - FP
    FN = true_edges - detected_edges

    confusion_matrix = np.array([[len(TN), len(FP)], [len(FN), len(TP)]])
    cm = confusion_matrix_networks(G_true, G_learned)
    assert_array_equal(confusion_matrix, cm)

    # normalize across columns (i.e. the predicted adjacencies)
    cm = confusion_matrix_networks(G_true, G_learned, normalize="pred")
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=0)
    assert_array_equal(confusion_matrix, cm)

    # normalize across columns (i.e. the true adjacencies)
    cm = confusion_matrix_networks(G_true, G_learned, normalize="true")
    confusion_matrix = np.array([[len(TN), len(FP)], [len(FN), len(TP)]])
    confusion_matrix = confusion_matrix.T / np.sum(confusion_matrix, axis=1)
    confusion_matrix = confusion_matrix.T
    assert_array_equal(confusion_matrix, cm)

    # now if we swap the labels ordering, then the confusion matrix is transposed
    cm = confusion_matrix_networks(G_true, G_learned, normalize="true", labels=[1, 0])
    assert_array_equal(confusion_matrix.T, cm)


def test_structure_hamming_dist():
    """Test structural hamming distance computation using graphs."""
    edges = [(0, 1)]
    nodes = [0, 1, 2]
    G = nx.DiGraph(edges)
    G.add_nodes_from(nodes)

    edges = [(0, 2)]
    test_G = nx.DiGraph(edges)
    test_G.add_nodes_from(nodes)

    # create an error
    with pytest.raises(RuntimeError, match="The type of graphs must be the same"):
        error_G = nx.Graph(edges)
        structure_hamming_dist(G, error_G)

    # compare the two graphs, which have two differing edges
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=False)
    assert shd == 2.0

    # adding the edge should reduce the distance
    G.add_edge(0, 2)
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=False)
    assert shd == 1.0

    # anticausal direction shouldn't matter
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=True)
    assert shd == 1.0

    # adding an edge in the wrong direction though will add double distance
    G.remove_edge(0, 2)
    G.add_edge(2, 0)
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=True)
    assert shd == 3.0
