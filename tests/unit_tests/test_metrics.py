import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal

from dodiscover.metrics import confusion_matrix_networks


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
