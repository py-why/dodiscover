import string

import networkx as nx
import numpy as np

from dodiscover import make_context
from dodiscover.metrics import structure_hamming_dist, toporder_divergence
from dodiscover.toporder.das import DAS
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import dummy_groundtruth, dummy_sample, full_dag, orders_consistency

# Fix the random seed
seed = 42


def test_given_dataset_when_fitting_DAS_then_shd_larger_equal_dtop():
    X = dummy_sample(seed=seed)
    G = dummy_groundtruth()
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.learn_graph(X, context)
    G_pred = model.graph_
    order_pred = model.order_
    shd = structure_hamming_dist(
        true_graph=G,
        pred_graph=G_pred,
        double_for_anticausal=False,
    )
    d_top = toporder_divergence(G, order_pred)
    assert shd >= d_top


def test_given_dag_and_dag_without_leaf_when_fitting_then_order_estimate_is_consistent():
    X = dummy_sample(seed=seed)
    order_gt = [2, 1, 3, 0]
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.learn_graph(X, context)
    order_full = model.order_
    model.learn_graph(X[order_gt[:-1]], context)
    order_noleaf = model.order_
    assert orders_consistency(order_full, order_noleaf)


def test_given_dataset_when_fitting_das_with_unit_pvalue_and_score_then_returns_equal_outputs():
    X = dummy_sample(seed=seed)
    context = make_context().variables(observed=X.columns).build()
    das = DAS(min_parents=0, das_cutoff=1)
    score = SCORE()
    das.learn_graph(X, context)
    A_das = nx.to_numpy_array(das.graph_)
    order_das = das.order_
    score.learn_graph(X, context)
    A_score = nx.to_numpy_array(score.graph_)
    order_score = score.order_
    assert order_das == order_score
    assert np.allclose(A_das, A_score)


def test_given_adjacency_when_fitting_then_returns_dag_with_context_included_edges():
    X = dummy_sample(seed=seed)
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.learn_graph(X, context)  # fit without context included edges
    A = nx.to_numpy_array(model.graph_)
    order = model.order_
    A_dense = full_dag(order)
    d = len(X.columns)
    edges = []  # include all edges in A_dense and not in A
    for i in range(d):
        for j in range(d):
            if A_dense[i, j] == 1 and A[i, j] == 0:
                edges.append((i, j))
    included_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    included_edges.add_edges_from(edges)
    context = make_context(context).edges(include=included_edges).build()
    model.learn_graph(X, context)  # fit with context included edges
    A_included = nx.to_numpy_array(model.graph_)
    assert np.allclose(A_dense, A_included)


def test_given_custom_nodes_labels_when_fitting_then_input_output_labels_are_consistent():
    X = dummy_sample(seed=seed)
    model = DAS()

    # Inference with default labels
    context_builder = make_context()
    context = context_builder.variables(observed=X.columns).build()
    model.learn_graph(X, context)
    A_default = nx.to_numpy_array(model.graph_)

    # Inference with custom labels
    labels = list(string.ascii_lowercase)[: len(X.columns)]
    X.columns = labels
    context_builder = make_context()
    context = context_builder.variables(observed=X.columns).build()
    model.learn_graph(X, context)
    A_custom = nx.to_numpy_array(model.graph_)

    assert list(model.graph_.nodes()) == labels  # check nodes have custom labels
    assert np.allclose(A_custom, A_default)  # check output not affected by relabeling
