import networkx as nx
import numpy as np
import pandas as pd
from pywhy_graphs import StationaryTimeSeriesDiGraph

from dodiscover.ci import Oracle
from dodiscover.constraint.timeseries import TimeSeriesPC
from dodiscover.constraint.timeseries.utils import convert_ts_df_to_multiindex
from dodiscover.context_builder import make_ts_context

seed = 12345
rng = np.random.default_rng(seed)


def test_evaluate_edge():
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
    alg = TimeSeriesPC(ci_estimator=oracle)
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


class TestTsPCSimple:
    """Test tsPC algorithm against the modified graph in the tsFCI paper."""

    def setup(self):
        pass

    def test_timeseries_pc_oracle(self):
        r"""Test tsPC's algorithm assuming no contemporaneous edges.

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
        alg = TimeSeriesPC(ci_estimator=oracle, separate_lag_phase=False, contemporaneous_edges=False)

        context = make_ts_context().max_lag(max_lag).variables(data=data).build()

        # learn the skeleton graph
        skel_graph, sep_set = alg.learn_skeleton(data, context)

        # all edges in skeleton are inside G
        assert all(edge in skel_graph.edges for edge in G.edges)
        assert nx.is_isomorphic(skel_graph.to_undirected(), G.to_undirected())
        assert sep_set[("x1", -1)][("x3", -1)] == []
        assert {("x3", -1)} in sep_set[("x2", 0)][("x3", 0)]

        # now test the full fit algorithm
        alg.fit(data, context)
        learned_graph = alg.graph_

        # all edges in skeleton are inside G
        assert nx.is_isomorphic(learned_graph.to_directed(), G.to_directed())


    def test_timeseries_pc_contemporaneous(self):
        """Test tsPC algorithm with contemporaneous edges.

        Uses figure 2 from PCMCI+ paper.
        """
        #    t-1   t
        # x   o -> o
        #     ^ \> ^
        # y   o -> o
        #     ^    ^
        # z   o    o
        var_names = ["x", "y", "z"]
        ts_edges = [
            (("x", -1), ("x", 0)),
            (("x", -1), ("y", 0)),
            (("y", -1), ("y", 0)),
            (("y", 0), ("x", 0)),
            (("z", 0), ("y", 0)),
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
        context = make_ts_context().max_lag(max_lag).variables(data=data).build()

        # First: test that the algorithm return the correct answer
        # learn the skeleton graph without contemporaneous edges
        alg = TimeSeriesPC(ci_estimator=oracle, separate_lag_phase=False, contemporaneous_edges=True)
        skel_graph, _ = alg.learn_skeleton(data, context)

        # all edges in skeleton are inside G
        assert all(edge in skel_graph.edges for edge in G.edges)
        assert nx.is_isomorphic(skel_graph.to_undirected(), G.to_undirected())
        # now test the full fit algorithm
        alg.fit(data, context)
        learned_graph = alg.graph_

        # all edges in skeleton are inside G
        assert nx.is_isomorphic(learned_graph.to_directed(), G.to_directed())
        assert nx.is_isomorphic(learned_graph.sub_directed_graph(), G)

        # learn the skeleton graph without contemporaneous edges
        alg = TimeSeriesPC(ci_estimator=oracle, separate_lag_phase=False, contemporaneous_edges=False)
        skel_graph, _ = alg.learn_skeleton(data, context)
        # all edges in skeleton are inside G
        assert not all(edge in skel_graph.edges for edge in G.edges)
        assert not nx.is_isomorphic(skel_graph.to_undirected(), G.to_undirected())


class TestTsPCSimulated:
    """Test tsPC algorithm against more complex simulated graphs."""

    def setup(self):
        pass

    