import networkx as nx
import pytest
from pywhy_graphs.simulate import simulate_random_er_dag

from dodiscover.datasets import make_linear_gaussian


@pytest.mark.parametrize("n_jobs", [None, -1])
def test_make_linear_gaussian_from_graph_n_jobs(n_jobs):
    G = simulate_random_er_dag(n_nodes=5, seed=12345, ensure_acyclic=True)

    G, data = make_linear_gaussian(G, random_state=12345, n_jobs=n_jobs)

    assert set(data.columns) == set(G.nodes)
    assert all(key in nx.get_node_attributes(G, "parent_functions") for key in G.nodes)
    assert all(key in nx.get_node_attributes(G, "gaussian_noise_function") for key in G.nodes)
