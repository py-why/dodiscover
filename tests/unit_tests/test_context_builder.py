import networkx as nx
import numpy as np
import pandas as pd

from dodiscover import make_context

seed = 12345


def make_df() -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    Y = rng.randn(300, 1)
    return pd.DataFrame(np.hstack((X, Y)), columns=["x", "y"])


def test_constructor():
    df = make_df()
    ctx = make_context(df)

    assert ctx is not None
    assert ctx.build().data is df


def test_build_with_initial_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "y")])
    ctx = make_context(make_df()).init_graph(graph).build()
    assert ctx.init_graph is graph


def test_build_with_observed_and_latents():
    ctx = make_context(make_df()).features(observed_variables=["x"], latent_variables=["y"]).build()
    assert ctx.observed_variables == set(["x"])
    assert ctx.latent_variables == set(["y"])
