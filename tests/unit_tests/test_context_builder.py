import networkx as nx

from dodiscover import make_context


def test_constructor():
    ctx = make_context()
    assert ctx is not None


def test_build_with_initial_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "y")])
    ctx = make_context().graph(graph).build()
    assert ctx.init_graph is graph


def test_build_with_observed_and_latents():
    ctx = make_context().variables(observed=["x"], latent=["y"]).build()
    assert ctx.observed_variables == set(["x"])
    assert ctx.latent_variables == set(["y"])
