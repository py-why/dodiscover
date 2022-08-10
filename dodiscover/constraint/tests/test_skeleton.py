import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dodiscover import Context
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover.ci.tests.testdata import bin_data, dis_data
from dodiscover.constraint import LearnSkeleton
from dodiscover.constraint.utils import dummy_sample


def common_cause_and_collider():
    """Test orienting a common cause and a collider.

    The following graph has some complexities to test the PC algorithm
    with the Oracle setting: ``1 <- 0 -> 2 <- 3``.
    """
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = nx.DiGraph(incoming_graph_data)
    return G


def collider():
    # build initial DAG
    edge_list = [
        ("x", "y"),
        ("z", "y"),
    ]
    G = nx.DiGraph(edge_list)
    return G


def complex_graph():
    """Complex graph from Figure 2 of :footcite:`Colombo2012`.

    Does not include the latent edges.

    References
    ----------
    .. footbibliography::
    """
    edge_list = [
        ("x4", "x1"),
        ("x2", "x5"),
        ("x3", "x2"),
        ("x3", "x4"),
        ("x2", "x6"),
        ("x3", "x6"),
        ("x4", "x6"),
        ("x5", "x6"),
    ]
    G = nx.DiGraph(edge_list)
    return G


@pytest.mark.parametrize(
    ("indep_test_func", "data_matrix", "g_answer"),
    [
        (
            GSquareCITest(),
            np.array(bin_data).reshape((5000, 5)),
            nx.DiGraph(
                {
                    0: (1,),
                    1: (),
                    2: (3, 4),
                    3: (1, 2),
                    4: (1, 2),
                }
            ),
        ),
        (
            GSquareCITest("discrete"),
            np.array(dis_data).reshape((10000, 5)),
            nx.DiGraph(
                {
                    0: (2,),
                    1: (2, 3),
                    2: (),
                    3: (),
                    4: (3,),
                }
            ),
        ),
    ],
)
def test_learn_skeleton_with_data(indep_test_func, data_matrix, g_answer, alpha=0.01):
    """Test PC algorithm for estimating the causal DAG."""
    data_df = pd.DataFrame(data_matrix)
    alg = LearnSkeleton(ci_estimator=indep_test_func, alpha=alpha)
    context = Context(data=data_df)
    alg.fit(context)

    # obtain the fitted skeleton graph
    skel_graph = alg.adj_graph_

    # all edges in the answer should be part of the skeleton graph
    for edge in g_answer.edges:
        error_msg = f"Edge {edge} should be in graph {skel_graph}"
        assert skel_graph.has_edge(*edge), error_msg


@pytest.mark.parametrize("G", [collider(), common_cause_and_collider(), complex_graph()])
def test_learn_skeleton_oracle(G):
    df = dummy_sample(G)
    oracle = Oracle(G)
    alpha = 0.05
    alg = LearnSkeleton(ci_estimator=oracle, alpha=alpha)
    context = Context(data=df)
    alg.fit(context)

    # obtain the fitted skeleton graph
    skel_graph = alg.adj_graph_

    # the skeleton of both graphs should match perfectly
    assert nx.is_isomorphic(skel_graph, G.to_undirected())
