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

    # if the initial graph does not match the variables passed in, then raise an error
    with pytest.raises(ValueError, match="The nodes within the initial graph*"):
        make_context().graph(graph).variables(observed="x").build()


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

    df = make_df()
    ctx_builder = make_context()
    # if we only set observed, then the latents should be inferred from the
    # dataset if there are any
    ctx = ctx_builder.variables(observed="x", data=df).build()
    assert ctx.latent_variables == {"y"}

    # if we only set latents, then the observed should be inferred from the dataset
    ctx = ctx_builder.variables(latents="x", data=df).build()
    assert ctx.observed_variables == {"y"}


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


def test_build_context_errors():
    ctx_builder = make_context()
    df = make_df()

    # variables should always be called
    with pytest.raises(ValueError, match="Could not infer variables from data"):
        ctx_builder.variables()
    with pytest.raises(ValueError, match="Could not infer variables from data"):
        ctx_builder.build()

    # if we specify latent and observed variables, they should match up with
    # the columns of the dataset
    with pytest.raises(ValueError, match="If observed and latents are set"):
        ctx_builder.variables(observed="x", latents="z", data=df)
