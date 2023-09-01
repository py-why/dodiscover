import string
import warnings

import networkx as nx
import numpy as np
import pytest
import torch
from dowhy.gcm.util.general import set_random_seed

from dodiscover import make_context
from dodiscover.continuous.grandag import GranDAG
from dodiscover.toporder.utils import dummy_groundtruth, dummy_sample

seed = 42  # Fix the random seed
num_train_iter = 1000  # Number of training iterations for unit tests


def fix_seeds(seed: int) -> None:
    set_random_seed(seed)
    torch.manual_seed(seed)


@pytest.mark.filterwarnings("ignore:h not converged")
def test_given_include_edges_then_raise_limited_support_warning():
    warning_message = (
        "Prior knowledge is only partially supported for this algorithm. Included edges may be "
        "removed during fitting (but not during final pruning step)!"
    )
    X = dummy_sample(seed=seed)
    G = dummy_groundtruth()
    model = GranDAG(num_train_iter=num_train_iter)
    context = make_context().variables(observed=X.columns).build()

    # Randomly pick one edge to include from the ones returned after fitting
    edges_idxs = np.where(nx.to_numpy_array(G))
    include_idx = np.random.randint(len(edges_idxs[0]))
    included_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    included_edges.add_edge(edges_idxs[0][include_idx], edges_idxs[1][include_idx])
    context = (
        make_context(context).edges(include=included_edges).build()
    )  # Update context with included edge

    with warnings.catch_warnings(record=True) as ws:
        model.fit(X, context)  # fit with context included edges
        assert ws and warning_message in [str(w.message) for w in ws]


@pytest.mark.filterwarnings("ignore:h not converged")
def test_given_include_edges_when_pns_then_returns_dag_with_included_edges():
    has_edges = False
    while not has_edges:
        X = dummy_sample(seed=seed)
        model = GranDAG(num_train_iter=num_train_iter, pns=False)  # Run without pns
        context = make_context().variables(observed=X.columns).build()
        model.fit(X, context)  # fit with context

        # Randomly pick one edge to include from the ones returned after fitting
        edges_idxs = np.where(nx.to_numpy_array(model.graph_))
        has_edges = not len(edges_idxs[0]) == 0
    include_idx = np.random.randint(len(edges_idxs[0]))
    included_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    included_edges.add_edge(edges_idxs[0][include_idx], edges_idxs[1][include_idx])
    model.context = (
        make_context(context).edges(include=included_edges).build()
    )  # Update context with included edge

    # Run pns with context with included edges
    adj = model.model.adjacency.detach().cpu().numpy()
    adj = model._prune(torch.tensor(X.to_numpy()), adj)
    model.graph_ = model._postprocess_output(nx.from_numpy_array(adj, create_using=nx.DiGraph))

    for edge in context.included_edges.edges():
        assert edge in model.graph_.edges()


@pytest.mark.filterwarnings("ignore:h not converged")
def test_given_exclude_edges_when_fitting_then_returns_dag_with_excluded_edges():
    X = dummy_sample(seed=seed)
    G = dummy_groundtruth()
    model = GranDAG(num_train_iter=num_train_iter)
    context = make_context().variables(observed=X.columns).build()

    # Randomly pick edge(s) to exclude
    no_edges_idxs = np.where(nx.to_numpy_array(G) == 0)
    exclude_idx = np.random.randint(len(no_edges_idxs[0]))

    excluded_edges = nx.empty_graph(len(X.columns), create_using=nx.DiGraph)
    excluded_edges.add_edge(no_edges_idxs[0][exclude_idx], no_edges_idxs[1][exclude_idx])

    context = make_context(context).edges(exclude=excluded_edges).build()
    model.fit(X, context)  # fit with context excluded edges

    for edge in context.excluded_edges.edges():
        assert edge not in model.graph_.edges()


@pytest.mark.filterwarnings("ignore:h not converged")
def test_given_dataset_and_dataset_with_permuted_column_when_fitting_then_return_equal_outputs():
    X = dummy_sample(seed=seed)
    model = GranDAG(num_train_iter=num_train_iter)
    context = make_context().variables(observed=X.columns).build()

    # permute sample columns
    permutation = [1, 3, 0, 2]
    permuted_sample = X[permutation]  # permute pd.DataFrame columns

    # Run inference on original and permuted data
    fix_seeds(seed=seed)
    model.fit(permuted_sample, context)
    A_permuted = nx.to_numpy_array(model.graph_)
    fix_seeds(seed=seed)
    model.fit(X, context)
    A = nx.to_numpy_array(model.graph_)

    # Match variables order
    back_permutation = [2, 0, 3, 1]
    A_permuted = A_permuted[:, back_permutation]
    A_permuted = A_permuted[back_permutation, :]

    assert np.allclose(A_permuted, A)


@pytest.mark.filterwarnings("ignore:h not converged")
def test_given_custom_nodes_labels_when_fitting_then_input_output_labels_are_consistent():
    X = dummy_sample(seed=seed)
    model = GranDAG(num_train_iter=num_train_iter)

    # Inference with default labels
    context_builder = make_context()
    context = context_builder.variables(observed=X.columns).build()
    fix_seeds(seed=seed)
    model.fit(X, context)
    A_default = nx.to_numpy_array(model.graph_)

    # Inference with custom labels
    labels = list(string.ascii_lowercase)[: len(X.columns)]
    X.columns = labels
    context_builder = make_context()
    context = context_builder.variables(observed=X.columns).build()
    fix_seeds(seed=seed)
    model.fit(X, context)
    A_custom = nx.to_numpy_array(model.graph_)

    assert list(model.graph_.nodes()) == labels  # check nodes have custom labels
    assert np.allclose(A_custom, A_default)  # check output not affected by relabeling
