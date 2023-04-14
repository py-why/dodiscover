import networkx as nx
import numpy as np
import pytest

from dodiscover import make_context
from dodiscover.metrics import structure_hamming_dist, toporder_divergence
from dodiscover.toporder.das import DAS
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import dummy_groundtruth, dummy_sample, full_dag, orders_consistency


@pytest.fixture
def seed():
    return 42


def test_given_dataset_when_fitting_DAS_then_shd_larger_equal_dtop(seed):
    X = dummy_sample(seed=seed)
    G = dummy_groundtruth()
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    G_pred = model.graph_
    order_pred = model.order_
    shd = structure_hamming_dist(
        true_graph=G,
        pred_graph=G_pred,
        double_for_anticausal=False,
    )
    d_top = toporder_divergence(G, order_pred)
    assert shd >= d_top


def test_given_dag_and_dag_without_leaf_when_fitting_then_order_estimate_is_consistent(seed):
    X = dummy_sample(seed=seed)
    order_gt = [2, 1, 3, 0]
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    order_full = model.order_
    model.fit(X[order_gt[:-1]], context)
    order_noleaf = model.order_
    assert orders_consistency(order_full, order_noleaf)


def test_given_dataset_and_rescaled_dataset_when_fitting_then_returns_equal_output(seed):
    X = dummy_sample(seed=seed)
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
    A = nx.to_numpy_array(model.graph_)
    model.fit(X * 2, context)
    A_rescaled = nx.to_numpy_array(model.graph_)
    assert np.allclose(A, A_rescaled)


def test_given_order_and_alternative_order_when_pruning_then_return_equal_outputs(seed):
    X = dummy_sample(seed=seed)
    order_gt = [2, 1, 3, 0]
    order_equivalent = [2, 3, 1, 0]
    model = DAS(min_parents=0)
    model.context = make_context().variables(observed=X.columns).build()
    model.var = [1 for _ in range(len(order_gt))]

    model.order_ = order_gt
    A_gt = model.prune(X.to_numpy(), full_dag(order_gt))

    model.order_ = order_equivalent
    A_equivalent = model.prune(X.to_numpy(), full_dag(order_equivalent))
    assert np.allclose(A_gt, A_equivalent)


def test_given_dataset_when_fitting_das_with_unit_pvalue_and_score_then_returns_equal_outputs(seed):
    X = dummy_sample(seed=seed)
    context = make_context().variables(observed=X.columns).build()
    das = DAS(min_parents=0, das_cutoff=1)
    score = SCORE()
    das.fit(X, context)
    A_das = nx.to_numpy_array(das.graph_)
    order_das = das.order_
    score.fit(X, context)
    A_score = nx.to_numpy_array(score.graph_)
    order_score = score.order_
    assert order_das == order_score
    assert np.allclose(A_das, A_score)


def test_given_adjacency_when_pruning_then_returns_dag_with_context_included_edges(seed):
    X = dummy_sample(seed=seed)
    model = DAS(min_parents=0)
    context = make_context().variables(observed=X.columns).build()
    model.fit(X, context)
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
    model.fit(X, context)
    A_included = nx.to_numpy_array(model.graph_)
    assert np.allclose(A_dense, A_included)
