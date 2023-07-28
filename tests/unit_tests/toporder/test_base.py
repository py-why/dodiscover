import networkx as nx
import numpy as np

from dodiscover import make_context
from dodiscover.toporder._base import CAMPruning, SteinMixin
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.utils import dummy_dense, dummy_sample, full_adj_to_order, full_dag

# Fix the random seed
seed = 42


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


def test_given_order_and_alternative_order_when_pruning_then_return_equal_outputs():
    X = dummy_sample(seed=seed)
    order_gt = [2, 1, 3, 0]
    order_equivalent = [2, 3, 1, 0]
    cam_pruning = CAMPruning()

    # Context information (nothing to encode)
    G_included = nx.empty_graph(create_using=nx.DiGraph)
    G_excluded = nx.empty_graph(create_using=nx.DiGraph)

    # Inference
    A_gt = cam_pruning.prune(X.to_numpy(), full_dag(order_gt), G_included, G_excluded)
    A_equivalent = cam_pruning.prune(
        X.to_numpy(), full_dag(order_equivalent), G_included, G_excluded
    )
    assert np.allclose(A_gt, A_equivalent)


def test_given_adjacency_when_pruning_then_excluded_edges_are_removed():
    G_dense = dummy_dense()
    X = dummy_sample(seed=seed)
    model = SCORE(alpha=1.0)  # do not remove edges with pruning

    # Get dense prediction without excluded edges
    context = make_context().variables(observed=X.columns).build()
    model.learn_graph(X, context)
    order = model.order_
    G_dense = model.graph_
    A_dense = nx.to_numpy_array(G_dense)

    # Exclude leaf node incoming edge
    leaf = order[-1]
    l_parents = np.argwhere(A_dense[:, leaf] == 1).squeeze(axis=1)
    excluded_edges = nx.DiGraph([(l_parents[0], leaf)])
    context = make_context().variables(data=X).edges(exclude=excluded_edges).build()
    model.learn_graph(X, context)
    G_excluded = model.graph_
    A_excluded = nx.to_numpy_array(G_excluded)
    assert A_excluded[l_parents[0], leaf] == 0


def test_given_adjacency_when_pruning_then_only_excluded_edges_are_removed():
    G_dense = dummy_dense()
    X = dummy_sample(seed=seed)
    model = SCORE(alpha=1.0)  # do not remove edges with pruning

    # Get dense prediction without excluded edges
    context = make_context().variables(observed=X.columns).build()
    model.learn_graph(X, context)
    order = model.order_
    G_dense = model.graph_
    A_dense = nx.to_numpy_array(G_dense)

    # Exclude leaf node incoming edge
    leaf = order[-1]
    l_parents = np.argwhere(A_dense[:, leaf] == 1).squeeze(axis=1)
    excluded_edges = nx.DiGraph([(l_parents[0], leaf)])
    context = make_context().variables(data=X).edges(exclude=excluded_edges).build()
    model.learn_graph(X, context)
    G_excluded = model.graph_
    A_excluded = nx.to_numpy_array(G_excluded)
    A_excluded[l_parents[0], leaf] = 1

    assert np.allclose(A_dense, A_excluded)


def test_given_adjacency_when_pruning_with_pns_then_excluded_edges_are_removed():
    G_dense = dummy_dense()
    X = dummy_sample(seed=seed)
    model = SCORE(alpha=1.0, pns=True)  # do not remove edges with pruning

    # Get dense prediction without excluded edges
    context = make_context().variables(observed=X.columns).build()
    model.learn_graph(X, context)
    order = model.order_
    G_dense = model.graph_
    A_dense = nx.to_numpy_array(G_dense)

    # Exclude leaf node incoming edge
    leaf = order[-1]
    l_parents = np.argwhere(A_dense[:, leaf] == 1).squeeze(axis=1)
    excluded_edges = nx.DiGraph([(l_parents[0], leaf)])
    context = make_context().variables(data=X).edges(exclude=excluded_edges).build()
    model.learn_graph(X, context)
    G_excluded = model.graph_
    A_excluded = nx.to_numpy_array(G_excluded)
    assert A_excluded[l_parents[0], leaf] == 0


# -------------------- Test SteinMixin -------------------- #
def test_given_dataset_when_fitting_the_hessian_then_hessian_is_symmetric():
    def check_symmetry(H):
        for row in range(len(H)):
            if not np.allclose(H[row], np.transpose(H[row], (1, 0))):
                return False
        return True

    X = dummy_sample(seed=seed)
    stein = SteinMixin()
    H = stein.hessian(X.to_numpy(), 0.001, 0.001)
    assert check_symmetry(H)
