import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dodiscover import make_context
from dodiscover.context_builder import make_ts_context

seed = 12345


def make_df() -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    Y = rng.randn(300, 1)
    return pd.DataFrame(np.hstack((X, Y)), columns=["x", "y"])


@pytest.mark.parametrize("ctx", [make_context(), make_ts_context()])
def test_constructor(ctx):
    assert ctx is not None


def test_build_with_initial_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "y")])
    data = make_df()
    ctx = make_context().init_graph(graph).variables(data=data).build()
    assert ctx.init_graph is graph


def test_build_with_observed_and_latents():
    ctx = make_context().variables(observed=set("x"), latents=set("y")).build()
    assert ctx.observed_variables == set("x")
    assert ctx.latent_variables == set("y")


def test_with_context():
    """Test make and builder with a previous context"""
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "y")])
    data = make_df()
    ctx = make_context().init_graph(graph).variables(data=data).build()

    new_ctx = make_context(ctx).build()

    # test equality
    assert ctx == new_ctx


def test_ts_context():
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "y")])
    data = make_df()
    max_lag = 2
    builder = make_ts_context().init_graph(graph).variables(data=data)

    with pytest.raises(ValueError, match="Max lag must be set before building time-series context"):
        builder.build()

    ctx = builder.max_lag(max_lag).build()
    new_ctx = make_ts_context(ctx).build()

    # test equality
    assert ctx == new_ctx
