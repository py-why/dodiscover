import dataclasses
import math
from copy import copy, deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dodiscover import ContextBuilder, InterventionalContextBuilder, make_context

seed = 12345


def make_df() -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    Y = rng.randn(300, 1)
    return pd.DataFrame(np.hstack((X, Y)), columns=["x", "y"])


def test_constructor():
    ctx = make_context()
    assert ctx is not None


def test_build_with_initial_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([("x", "y")])
    data = make_df()
    ctx = make_context().init_graph(graph).variables(data=data).build()
    assert ctx.init_graph is graph

    # if the initial graph does not match the variables passed in, then raise an error
    with pytest.raises(ValueError, match="The nodes within the initial graph*"):
        make_context().init_graph(graph).variables(observed="blah").build()

    # this should work since 'x' is in the subset of the initial graph
    ctx = make_context().init_graph(graph).variables(observed="x").build()


def test_build_with_observed_and_latents():
    ctx = make_context().variables(observed=set("x"), latents=set("y")).build()
    assert ctx.observed_variables == set("x")
    assert ctx.latent_variables == set("y")

    df = make_df()
    ctx_builder = make_context()
    # if we only set observed, then the latents should be inferred from the
    # dataset if there are any
    ctx = ctx_builder.variables(observed="x", data=df).build()
    assert ctx.latent_variables == {"y"}

    # if we only set latents, then the observed should be inferred from the dataset
    ctx = ctx_builder.variables(latents="x", data=df).build()
    assert ctx.observed_variables == {"y"}

    cp_ctx_builder = make_context()
    with pytest.raises(RuntimeError, match="Observed variables are set already"):
        cp_ctx_builder.observed_variables({"x"})
        cp_ctx_builder.latent_variables({"x"})

    cp_ctx_builder = deepcopy(ctx_builder)
    with pytest.raises(RuntimeError, match="Latent variables are set already"):
        cp_ctx_builder.latent_variables({"x"})
        cp_ctx_builder.observed_variables({"x"})


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
    with pytest.raises(ValueError, match="If observed and latents are both set"):
        ctx_builder.variables(observed="x", latents="z", data=df)


# def test_context_set_errors():
#     ctx_builder = make_context()
#     df = make_df()
#     ctx = ctx_builder.variables(data=df).build()
# with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
#     ctx.init_graph = nx.empty_graph(0)


def test_context_set_edges():
    ctx_builder = make_context()
    df = make_df()

    # an error should be raised in inclusion/exclusion edges do not match
    inc_graph = nx.Graph()
    inc_graph.add_edge("x", "y")
    ctx_builder.variables(data=df).included_edges(inc_graph)
    with pytest.raises(RuntimeError, match="^(.*)is already specified as an included edge"):
        ctx_builder.excluded_edges(inc_graph)

    inc_graph = nx.Graph()
    ctx_builder = ctx_builder.included_edges(None)
    inc_graph.add_edge("x", "y")
    ctx_builder.excluded_edges(inc_graph)

    with pytest.raises(RuntimeError, match="^(.*) is already specified as an excluded edge"):
        ctx_builder.included_edges(inc_graph)


class BadContextBuilder(ContextBuilder):
    random_attribute: str = "hi"


def test_context_builder_extension_error():
    """All context builders should follow a specific pattern for definine private attributes."""

    with pytest.raises(RuntimeError, match="Context objects has class attributes that do not have"):
        BadContextBuilder().observed_variables(["x"]).build()


def test_context_set_get():
    ctx_builder = make_context()
    df = make_df()
    ctx = (
        ctx_builder.variables(data=df)
        .init_graph(nx.empty_graph(df.columns))
        .included_edges(nx.DiGraph([("x", "y")]))
        .build()
    )

    # making the contexBuilder with that context should result in the exact same copy
    ctx2 = make_context(ctx).build()
    assert ctx == ctx2

    # directly setting fields should not be allowed
    # with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
    #     ctx.intervention_targets = ["new"]

    # however, altering via functions is fine
    ctx.add_state_variable("new", 0)
    assert ctx.state_variables == {"new": 0}

    # basic smoke-check of functionality of dataclasses
    assert all(
        x not in dataclasses.asdict(ctx) for x in ("get_f_nodes", "get_params", "get_sigma_map")
    )

    ctx_copy = ctx.copy()
    assert ctx == ctx_copy


def test_context_state_variables():
    ctx_builder = make_context()
    df = make_df()
    ctx = (
        ctx_builder.variables(data=df)
        .init_graph(nx.empty_graph(df.columns))
        .included_edges(nx.DiGraph([("x", "y")]))
        .build()
    )

    with pytest.raises(RuntimeError, match="^(.*) is not a state variable:"):
        ctx.state_variable("pag")
    with pytest.warns(UserWarning, match="^(.*) is not a state variable:"):
        ctx.state_variable("pag", on_missing="warn")
    assert ctx.state_variable("pag", on_missing="ignore") is None


def test_context_interventions():
    ctx_builder = make_context(create_using=InterventionalContextBuilder)
    df = make_df()

    # check InterventionalContextBuilder errors that should be raised
    with pytest.warns(UserWarning, match="There is no intervention context set"):
        copy(ctx_builder).variables(data=df).init_graph(nx.empty_graph(df.columns)).build()

    with pytest.raises(RuntimeError, match="Not all nodes"):
        copy(ctx_builder).variables(data=df).init_graph(
            nx.empty_graph(list(df.columns) + ["blah"])
        ).num_distributions(5).build()

    with pytest.raises(RuntimeError, match="Setting the number of distribution"):
        copy(ctx_builder).variables(data=df).intervention_targets([("x",)]).num_distributions(
            5
        ).build()

    with pytest.raises(RuntimeError, match="Setting the number of distribution"):
        copy(ctx_builder).variables(data=df).num_distributions(5).intervention_targets(
            [("x",)]
        ).build()

    # check InterventionalContextBuilder building with
    # known-interventional targets
    # now build context
    ctx = (
        copy(ctx_builder)
        .variables(data=df)
        .intervention_targets([("x",), ("x", "y"), ("y",)])
        .build()
    )
    assert ctx.obs_distribution is True
    assert len(ctx.intervention_targets) == 3
    expected_num_f_nodes = math.comb(4, 2)
    assert len(ctx.sigma_map) == expected_num_f_nodes
    assert len(ctx.f_nodes) == expected_num_f_nodes
    assert ctx.symmetric_diff_map.keys() == ctx.sigma_map.keys()
    for val in [("x",), ("x", "y"), ("y",)]:
        assert frozenset(val) in ctx.symmetric_diff_map.values()

    # check that there are (4 choose 2) = 6 F-nodes
    assert len(ctx.f_nodes) == 6

    # now build context by also specifying number of distributions should be the same
    ctx2 = (
        copy(ctx_builder)
        .variables(data=df)
        .num_distributions(4)
        .intervention_targets([("x",), ("x", "y"), ("y",)])
        .build()
    )
    assert ctx == ctx2

    # making a copy should not change anything
    ctx3 = make_context(ctx, create_using=InterventionalContextBuilder).build()
    assert ctx == ctx3

    # now build context without intervention targets
    ctx = copy(ctx_builder).variables(data=df).num_distributions(4).build()

    # the symmetric diff map is now empty because we do not know the targets
    assert ctx.symmetric_diff_map == dict()
    assert set(ctx.sigma_map.keys()) == set(ctx.f_nodes)

    # test reverse sigma map
    assert set(ctx.reverse_sigma_map().values()) == set(ctx.f_nodes)


def test_context_interventions_without_observational():
    ctx_builder = make_context(create_using=InterventionalContextBuilder)
    df = make_df()

    # now build context without observational data
    ctx = (
        copy(ctx_builder)
        .variables(data=df)
        .obs_distribution(False)
        .num_distributions(3)
        .intervention_targets([("x",), ("x", "y"), ("y",)])
        .build()
    )

    # check that there are (3 choose 2) = 3 F-nodes
    assert len(ctx.f_nodes) == 3
