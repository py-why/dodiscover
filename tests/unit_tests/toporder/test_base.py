import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dodiscover import make_context
from dodiscover.toporder._base import SteinMixin
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import dummy_dense, dummy_sample, full_adj_to_order, full_dag


@pytest.fixture
def seed():
    return 42


def test_given_fully_connected_adjacency_when_applying_full_adj_to_order_then_order_is_correct():
    G_dense = dummy_dense()
    order = full_adj_to_order(nx.to_numpy_array(G_dense))
    assert order == [2, 1, 3, 0]


def test_given_order_when_applying_full_dag_then_fully_connected_adjacency_is_correct():
    """
    Test full_dag() function mapping an order to its unique
    adjacency matrix representation
    """
    G_dense = dummy_dense()
    order = [2, 1, 3, 0]
    assert np.allclose(full_dag(order), nx.to_numpy_array(G_dense))


def test_given_order_and_alternative_order_when_pruning_then_return_equal_outputs(seed):
    X = dummy_sample(seed=seed)
    order_gt = [2, 1, 3, 0]
    order_equivalent = [2, 3, 1, 0]
    model = SCORE()
    model.context = make_context().variables(observed=X.columns).build()

    model.order_ = order_gt
    A_gt = model.prune(X.to_numpy(), full_dag(order_gt))

    model.order_ = order_equivalent
    A_equivalent = model.prune(X.to_numpy(), full_dag(order_equivalent))
    assert np.allclose(A_gt, A_equivalent)


def test_given_adjacency_when_pruning_then_excluded_edges_are_removed(seed):
    G_dense = dummy_dense()
    X = dummy_sample(seed=seed)
    model = SCORE()
    model.context = make_context().variables(observed=X.columns).build()
    A_dense = nx.to_numpy_array(G_dense)
    order = full_adj_to_order(A_dense)
    model.order_ = order
    data = X.to_numpy()
    A = model.prune(data, A_dense)  # find prediction without excluded edges

    # Exclude leaf node incoming edge
    leaf = order[-1]
    l_parents = np.argwhere(A[:, leaf] == 1).squeeze(axis=1)
    excluded_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    excluded_edges.add_edges_from([(l_parents[0], leaf)])
    model.context = make_context(model.context).edges(exclude=excluded_edges).build()
    A_excluded = model.prune(data, A_dense)
    assert A_excluded[l_parents[0], leaf] == 0


def test_given_adjacency_when_pruning_with_pns_then_excluded_edges_are_removed(seed):
    G_dense = dummy_dense()
    X = dummy_sample(seed=seed)
    model = SCORE(pns=True)
    model.context = make_context().variables(observed=X.columns).build()
    A_dense = nx.to_numpy_array(G_dense)
    order = full_adj_to_order(A_dense)
    model.order_ = order
    data = X.to_numpy()
    A = model.prune(data, A_dense.astype(np.float_))  # find prediction without excluded edges

    # Exclude leaf node incoming edge
    leaf = order[-1]
    l_parents = np.argwhere(A[:, leaf] == 1).squeeze(axis=1)
    excluded_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    excluded_edges.add_edges_from([(l_parents[0], leaf)])
    model.context = make_context(model.context).edges(exclude=excluded_edges).build()
    A_excluded = model.prune(data, A_dense.astype(np.float_))
    assert A_excluded[l_parents[0], leaf] == 0


# -------------------- Test SteinMixin -------------------- #
def test_given_dataset_when_fitting_the_hessian_then_hessian_is_symmetric(seed):
    def check_symmetry(H):
        for row in range(len(H)):
            if not np.allclose(H[row], np.transpose(H[row], (1, 0))):
                return False
        return True

    X = dummy_sample(seed=seed)
    stein = SteinMixin()
    H = stein.hessian(X.to_numpy(), 0.001, 0.001)
    assert check_symmetry(H)
