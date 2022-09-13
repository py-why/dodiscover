import networkx as nx
import numpy as np
import pandas as pd

from dodiscover import make_context

seed = 12345


def make_df():
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    Y = rng.randn(300, 1)
    Z = rng.randn(300, 1)
    return pd.DataFrame(np.hstack((X, Y, Z)), columns=["x", "y", "z"])


def test_constructor():
    df = make_df()
    ctx = make_context(df)

    assert ctx is not None
    assert ctx.data == df
    assert ctx.build().data == df


def test_build_with_initial_graph():
    graph = nx.DiGraph()
    ctx = make_context(make_df()).init_graph(graph)
    built = ctx.build()
    assert built.graph == graph


def test_build_with_observed_and_latents():
    ctx = make_context(make_df()).features(observed=["x"], latents=["y"])
    built = ctx.build()
    assert built.observed == ["x"]
    assert built.latents == ["y"]
