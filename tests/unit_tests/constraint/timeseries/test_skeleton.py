import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pywhy_graphs import StationaryTimeSeriesDiGraph, StationaryTimeSeriesGraph

from dodiscover.ci import Oracle
from dodiscover.constraint.timeseries import LearnTimeSeriesSkeleton
from dodiscover.constraint.timeseries.utils import convert_ts_df_to_multiindex
from dodiscover.context_builder import make_ts_context

seed = 12345
rng = np.random.default_rng(seed)


def test_skeleton_evaluate_edge():
    var_names = ["x1", "x2", "x3"]
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x1", 0), ("x3", 0)),
    ]
    max_lag = 1
    G = StationaryTimeSeriesDiGraph(max_lag=max_lag)
    G.add_edges_from(ts_edges)

    # create a dataset, starting from the observed time-series
    data_arr = rng.random((10, len(var_names))).round(2)
    data = pd.DataFrame(
        data=data_arr,
        columns=var_names,
    )

    # create an oracle
    oracle = Oracle(G)
    alg = LearnTimeSeriesSkeleton(ci_estimator=oracle)
    data = convert_ts_df_to_multiindex(data, max_lag)

    _, pvalue = alg.evaluate_edge(data, ("x1", 0), ("x1", -1))
    assert pvalue == 0.0

    _, pvalue = alg.evaluate_edge(data, ("x1", 0), ("x2", 0))
    assert pvalue == 0.0

    _, pvalue = alg.evaluate_edge(data, ("x1", 0), ("x2", 0), {("x1", -1)})
    assert pvalue == 1.0

    _, pvalue = alg.evaluate_edge(data, ("x3", -1), ("x1", 0))
    assert pvalue == 0.0

    _, pvalue = alg.evaluate_edge(data, ("x3", -1), ("x1", 0), {("x1", -1)})
    assert pvalue == 1.0


def test_markovian_skeleton_oracle():
    r"""Test tsPC's skeleton algorithm  assuming no latent confounders nor contemporaneous edges.

    Tests the ts skeleton method with an oracle from
    Figure 1 and 2 of the tsFCI paper with the difference that the graph
    is fully observed here.

    Figure 1, where "\>" and "/>" are edges pointing
    down, or up diagonally.

    x1(t-2) -> x1(t-1) -> x1(t)
            \>          \>
    x2(t-2)    x2(t-1)    x2(t)
            />          />
    x3(t-2) -> x3(t-1) -> x3(t)
    """
    var_names = ["x1", "x2", "x3"]
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
    ]
    max_lag = 1
    G = StationaryTimeSeriesDiGraph(max_lag=max_lag)
    G.add_edges_from(ts_edges)

    # create a dataset, starting from the observed time-series
    data_arr = rng.random((10, len(var_names))).round(2)
    data = pd.DataFrame(
        data=data_arr,
        columns=var_names,
    )

    # create an oracle
    oracle = Oracle(G)
    alg = LearnTimeSeriesSkeleton(
        ci_estimator=oracle, separate_lag_phase=False, contemporaneous_edges=False
    )

    context = make_ts_context().max_lag(max_lag).variables(data=data).build()

    for edge in G.edges:
        print(edge)
    print(G.nodes)
    alg.fit(data, context)
    skel_graph = alg.adj_graph_

    # all edges in skeleton are inside G
    assert all(edge in skel_graph.edges for edge in G.edges)

    for edge in skel_graph.edges:
        print(edge)
    assert nx.is_isomorphic(skel_graph.to_undirected(), G.to_undirected())


def test_markovian_skeleton_with_contemporaneous_edges():
    r"""Test tsFCI's "PC" algorithm skeleton assuming contemporaneous edges.

    Tests the ts skeleton method with an oracle from a modified version of
    Figure 1 and 2 of the tsFCI paper.

    Figure 1, where "\>" and "/>" are edges pointing
    down, or up diagonally.

    x1(t-2) -> x1(t-1) -> x1(t)
            \>          \>
    x2(t-2)    x2(t-1)    x2(t)
            />          />
    x3(t-2) -> x3(t-1) -> x3(t)

    with contemporaneous edges
    x1(t) -> x2(t);
    x3(t) -> x2(t)
    """
    var_names = ["x1", "x2", "x3"]
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x1", 0), ("x3", 0)),
        (("x3", 0), ("x2", 0)),
    ]
    max_lag = 1
    G = StationaryTimeSeriesDiGraph(max_lag=max_lag)
    G.add_edges_from(ts_edges)

    # create a dataset, starting from the observed time-series
    data_arr = rng.random((10, len(var_names))).round(2)
    data = pd.DataFrame(
        data=data_arr,
        columns=var_names,
    )

    # create an oracle
    oracle = Oracle(G)
    alg = LearnTimeSeriesSkeleton(ci_estimator=oracle)
    context = make_ts_context().max_lag(max_lag).variables(data=data).build()

    # learn the graph
    alg.fit(data, context)
    skel_graph = alg.adj_graph_

    # all edges in skeleton are inside G
    assert all(edge in skel_graph.edges for edge in G.edges)
    assert nx.is_isomorphic(skel_graph.to_undirected(), G.to_undirected())


@pytest.mark.skip(reason="make work...")
def test_semi_markovian_skeleton_oracle():
    r"""Test tsFCI's "FCI" algorithm skeleton assuming latent confounders.

    Tests the ts skeleton method with an oracle from
    Figure 1 and 2 of the tsFCI paper.

    Figure 1, where "\>" and "/>" are edges pointing
    down, or up diagonally.

    x1(t-2) -> x1(t-1) -> x1(t)
            \>          \>
    x2(t-2)    x2(t-1)    x2(t)
            />          />
    x3(t-2) -> x3(t-1) -> x3(t)

    However, in the dataset, x3 is not observed. The expected
    skeleton is the undirected graph version of Figure 2b.

    x1(t-2)  -> x1(t-1) -> x1(t)
       ^
       |    \           \
       v     >           >
    x2(t-2) <-> x2(t-1) <-> x2(t)

    and x2(t-2) <-> x2(t)
    """
    var_names = ["x1", "x2", "x3"]
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
    ]
    max_lag = 2
    G = StationaryTimeSeriesDiGraph(max_lag=max_lag)
    G.add_edges_from(ts_edges)

    # create expected graph
    expected_G = StationaryTimeSeriesGraph(max_lag=max_lag)
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x2", -1), ("x2", 0)),
        (("x2", -2), ("x2", 0)),
        (("x1", -2), ("x2", -2)),  # contemporaneous edge at end
    ]
    expected_G.add_edges_from(ts_edges)

    # create a dataset, starting from the observed time-series
    data_arr = rng.random((10, len(var_names))).round(2)
    data = pd.DataFrame(
        data=data_arr,
        columns=var_names,
    )
    data.drop("x3", axis=1, inplace=True)

    # create an oracle using the original graph``
    oracle = Oracle(G)
    alg = LearnTimeSeriesSkeleton(
        ci_estimator=oracle,
        # separate_lag_phase=separate_lag_phase,
        latent_confounding=True,
    )

    context = make_ts_context().max_lag(max_lag).variables(data=data).build()
    alg.fit(data, context)
    skel_graph = alg.adj_graph_

    for edge in skel_graph.edges:
        print(sorted(edge, key=lambda x: x[1]))

    # all edges in skeleton are inside G
    assert all(edge in skel_graph.edges for edge in G.edges)
    assert nx.is_isomorphic(skel_graph.to_undirected(), expected_G.to_undirected())
