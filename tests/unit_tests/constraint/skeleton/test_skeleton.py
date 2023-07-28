import networkx as nx
import numpy as np
import pandas as pd
import pytest
import pywhy_graphs

from dodiscover import ContextBuilder, make_context
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover.constraint.skeleton import LearnSemiMarkovianSkeleton, LearnSkeleton
from dodiscover.constraint.utils import dummy_sample
from dodiscover.testdata.testdata import bin_data, dis_data


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
def test_learn_skeleton_with_data(indep_test_func, data_matrix, g_answer):
    """Test PC algorithm for estimating the causal DAG."""
    data_df = pd.DataFrame(data_matrix)
    alg = LearnSkeleton(ci_estimator=indep_test_func)
    context = make_context().variables(data=data_df).build()
    alg.learn_graph(data_df, context)

    # obtain the fitted skeleton graph
    skel_graph = alg.adj_graph_

    # all edges in the answer should be part of the skeleton graph
    for edge in g_answer.edges:
        error_msg = f"Edge {edge} should be in graph {skel_graph}"
        assert skel_graph.has_edge(*edge), error_msg


@pytest.mark.parametrize("skel_method", [LearnSkeleton, LearnSemiMarkovianSkeleton])
@pytest.mark.parametrize("G", [collider(), common_cause_and_collider(), complex_graph()])
def test_learn_skeleton_oracle(G, skel_method):
    df = dummy_sample(G)
    oracle = Oracle(G)
    alpha = 0.05
    alg = skel_method(ci_estimator=oracle, alpha=alpha)
    context = make_context().variables(data=df).build()
    alg.learn_graph(df, context)

    # obtain the fitted skeleton graph
    skel_graph = alg.adj_graph_

    # the skeleton of both graphs should match perfectly
    assert nx.is_isomorphic(skel_graph, G.to_undirected())


@pytest.mark.parametrize("skel_method", [LearnSkeleton, LearnSemiMarkovianSkeleton])
def test_method_does_not_change_context(skel_method):
    # reconstruct the PAG the way FCI would
    edge_list = [("D", "A")]
    latent_edge_list = [("A", "B"), ("D", "E")]
    graph = pywhy_graphs.ADMG(
        incoming_directed_edges=edge_list, incoming_bidirected_edges=latent_edge_list
    )
    ci_estimator = Oracle(graph)
    sample = dummy_sample(graph)
    context = make_context().variables(data=sample).build()
    context_copy = context.copy()

    # after the first stage, we learn a skeleton as in Figure 16
    firstalg = skel_method(ci_estimator=ci_estimator)
    firstalg.learn_graph(sample, context)

    # context should not change as a copy is made internally
    # assert context == context_copy
    # assert nx.is_isomorphic(context.init_graph, context_copy.init_graph)
    assert nx.is_isomorphic(context.included_edges, context_copy.included_edges)
    assert nx.is_isomorphic(context.excluded_edges, context_copy.excluded_edges)


def test_learn_pds_skeleton():
    """Test example in Causation, Prediction and Search book.

    See book Figure 16

    See: https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/.g/scottd/fullbook.pdf
    """  # noqa
    # reconstruct the PAG the way FCI would
    edge_list = [("D", "A"), ("B", "E"), ("F", "B"), ("C", "F"), ("C", "H"), ("H", "D")]
    latent_edge_list = [("A", "B"), ("D", "E")]
    graph = pywhy_graphs.ADMG(
        incoming_directed_edges=edge_list, incoming_bidirected_edges=latent_edge_list
    )
    ci_estimator = Oracle(graph)
    sample = dummy_sample(graph)
    context = make_context().variables(data=sample).build()

    # after the first stage, we learn a skeleton as in Figure 16
    firstalg = LearnSkeleton(ci_estimator=ci_estimator)
    firstalg.learn_graph(sample, context)
    pag_graph = pywhy_graphs.PAG(incoming_circle_edges=firstalg.adj_graph_)

    # generate the expected PAG
    edge_list = [
        ("A", "B"),
        ("D", "A"),
        ("B", "E"),
        ("B", "F"),
        ("F", "C"),
        ("C", "H"),
        ("H", "D"),
        ("D", "E"),
        ("A", "E"),  # Note: this is the extra edge
    ]
    expected_skel = nx.Graph(edge_list)
    assert nx.is_isomorphic(expected_skel, pag_graph.to_undirected())
    assert nx.is_isomorphic(expected_skel, firstalg.adj_graph_)

    # now, we will run the second stage of learning the skeleton with the PDS set
    # we should now learn Figure 17
    edge_list = [
        ("D", "A"),
        ("B", "E"),
        ("H", "D"),
        ("F", "B"),
    ]
    latent_edge_list = [("A", "B"), ("D", "E")]
    uncertain_edge_list = [
        ("A", "E"),
        ("A", "D"),
        ("E", "B"),
        ("E", "A"),
        ("B", "F"),
        ("F", "C"),
        ("C", "F"),
        ("C", "H"),
        ("H", "C"),
        ("D", "H"),
    ]
    exp_first_stage_pag = pywhy_graphs.PAG(
        edge_list,
        incoming_bidirected_edges=latent_edge_list,
        incoming_circle_edges=uncertain_edge_list,
    )

    # learn the skeleton of the graph now with the first stage skeleton
    alg = LearnSemiMarkovianSkeleton(ci_estimator=ci_estimator)
    alg.learn_graph(sample, context)

    # the first stage PAG should be equal to what we learned
    first_stage_pag = alg.context_.state_variable("PAG")
    for edge_type, subgraph in exp_first_stage_pag.get_graphs().items():
        assert nx.is_isomorphic(first_stage_pag.get_graphs(edge_type), subgraph)

    # now test that the expected skeleton is the same as the learned skeleton
    skel_graph = alg.adj_graph_
    edge_list = [
        ("D", "A"),
        ("B", "E"),
        ("H", "D"),
        ("F", "B"),
    ]
    latent_edge_list = [("A", "B"), ("D", "E")]
    uncertain_edge_list = [
        ("B", "F"),
        ("F", "C"),
        ("C", "F"),
        ("C", "H"),
        ("H", "C"),
        ("D", "H"),
    ]
    expected_pag = pywhy_graphs.PAG(
        edge_list,
        incoming_bidirected_edges=latent_edge_list,
        incoming_circle_edges=uncertain_edge_list,
    )
    expected_skel = expected_pag.to_undirected()
    for edge in expected_skel.edges():
        if not skel_graph.has_edge(*edge):
            print(f"Missing edge: {edge}")
    for edge in skel_graph.edges():
        if not expected_skel.has_edge(*edge):
            print(f"Extra edge: {edge}")
    assert nx.is_isomorphic(expected_skel, skel_graph)


def test_learn_skeleton_pds_disabled_first_stage():
    """Test that we can disable the first stage of the algorithm."""
    # reconstruct the PAG the way FCI would
    edge_list = [("D", "A"), ("B", "E"), ("F", "B"), ("C", "F"), ("C", "H"), ("H", "D")]
    latent_edge_list = [("A", "B"), ("D", "E")]
    graph = pywhy_graphs.ADMG(
        incoming_directed_edges=edge_list, incoming_bidirected_edges=latent_edge_list
    )
    ci_estimator = Oracle(graph)
    sample = dummy_sample(graph)
    context = (
        make_context(create_using=ContextBuilder).variables(data=sample).state_variables({}).build()
    )

    # generate the expected PAG
    edge_list = [
        ("A", "B"),
        ("D", "A"),
        ("B", "E"),
        ("B", "F"),
        ("F", "C"),
        ("C", "H"),
        ("H", "D"),
        ("D", "E"),
        ("A", "E"),  # Note: this is the extra edge
    ]
    expected_skel = nx.Graph(edge_list)

    # learn the skeleton of the graph now with the first stage skeleton
    alg = LearnSemiMarkovianSkeleton(
        ci_estimator=ci_estimator, second_stage_condsel_method=None, n_jobs=1
    )
    alg.learn_graph(sample, context)
    assert alg.context_.state_variable("PAG", on_missing="ignore") is None
    assert nx.is_isomorphic(expected_skel, alg.adj_graph_)
