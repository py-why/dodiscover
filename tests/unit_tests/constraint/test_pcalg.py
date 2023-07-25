import networkx as nx
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pywhy_graphs import ADMG, CPDAG

from dodiscover import make_context
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover.constraint import PC
from dodiscover.constraint.config import ConditioningSetSelection
from dodiscover.constraint.utils import dummy_sample
from dodiscover.metrics import confusion_matrix_networks
from dodiscover.testdata.testdata import bin_data, dis_data


@pytest.mark.parametrize(
    "pc_kwargs",
    [
        {
            "min_cond_set_size": 0,
            "max_cond_set_size": 3,
            "max_combinations": 10,
            "max_iter": 10,
            "condsel_method": ConditioningSetSelection.NBRS_PATH,
        },
        {},
    ],
)
@pytest.mark.parametrize(
    ("indep_test_func", "data_matrix", "g_answer", "alpha"),
    [
        (
            GSquareCITest(),
            np.array(bin_data).reshape((5000, 5)),
            nx.DiGraph(
                {
                    0: (1,),
                    1: (),
                    2: (),
                    3: (1,),
                    4: (1,),
                }
            ),
            0.01,
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
            0.1,  # Note: that alpha level of >= 0.1 is required for 2 and 3 to be dependent
        ),
    ],
)
def test_estimate_cpdag_testdata(indep_test_func, data_matrix, g_answer, alpha, pc_kwargs):
    """Test PC algorithm for estimating the causal DAG.

    Test data is imported from pcalg, which is sensitive to the CI test
    and alpha level. This illustrates an implicit tradeoff for
    structure learning algorithms, which may get the wrong answer due
    to finite samples.
    """
    data_df = pd.DataFrame(data_matrix)
    context = make_context().variables(data=data_df).build()
    alg = PC(ci_estimator=indep_test_func, alpha=alpha, **pc_kwargs)
    alg.learn_graph(data_df, context)
    graph = alg.graph_

    error_msg = "True edges should be: %s" % (g_answer.edges,)
    assert nx.is_isomorphic(graph.sub_directed_graph(), g_answer), error_msg

    # Test confusion matrix
    cm = confusion_matrix_networks(graph, g_answer)

    # The total number of edges if we assume symmetric graph: (N^2 - N) / 2
    ub_num_edges = (len(graph.nodes) ** 2 - len(graph.nodes)) / 2

    # now construct expected confusion matrix
    expected_cm = np.diag(
        [
            ub_num_edges - len(graph.directed_edges) - len(graph.undirected_edges),
            len(graph.directed_edges),
        ]
    )
    expected_cm[1, 0] = len(graph.undirected_edges)
    assert_array_equal(cm, expected_cm)

    # test what happens if fixed edges are present
    fixed_edges = nx.complete_graph(data_df.columns.values)
    context = make_context().variables(data=data_df).edges(include=fixed_edges).build()
    alg = PC(ci_estimator=indep_test_func, alpha=alpha)
    alg.learn_graph(data_df, context)
    complete_graph = alg.graph_
    assert nx.is_isomorphic(complete_graph.sub_undirected_graph(), fixed_edges)
    assert not nx.is_isomorphic(complete_graph.sub_directed_graph(), g_answer)


def test_common_cause_and_collider():
    """Test orienting a common cause and a collider.

    The following graph has some complexities to test the PC algorithm
    with the Oracle setting: ``1 <- 0 -> 2 <- 3``.
    """
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = nx.DiGraph(incoming_graph_data)
    df = dummy_sample(G)
    context = make_context().variables(data=df).build()
    pc = PC(ci_estimator=Oracle(G), apply_orientations=True)
    pc.learn_graph(df, context)
    cpdag = pc.graph_

    # compare with the expected CPDAG
    expected_cpdag = CPDAG(incoming_undirected_edges=G)
    expected_cpdag.orient_uncertain_edge(3, 2)
    expected_cpdag.orient_uncertain_edge(0, 2)
    assert nx.is_isomorphic(cpdag.sub_directed_graph(), expected_cpdag.sub_directed_graph())
    assert nx.is_isomorphic(cpdag.sub_undirected_graph(), expected_cpdag.sub_undirected_graph())


def test_collider():
    # construct a causal graph that will result in
    # x -> y <- z
    G = nx.DiGraph([("x", "y"), ("z", "y")])

    # construct the SCM and the corresponding causal graph
    oracle = Oracle(G)
    pc = PC(ci_estimator=oracle)
    sample = dummy_sample(G)
    context = make_context().variables(data=sample).build()
    pc.learn_graph(sample, context)
    graph = pc.graph_

    assert graph.has_edge("x", "y", graph.directed_edge_name)
    assert graph.has_edge("z", "y", graph.directed_edge_name)


def test_pc_rule1():
    edges = [
        ("x", "y"),
        ("z", "y"),
    ]
    G = ADMG(edges)
    oracle = Oracle(G)
    pc = PC(ci_estimator=oracle)

    # If C -> A - B, such that the triple is unshielded,
    # then orient A - B as A -> B

    G = CPDAG()
    G.add_edge("A", "B", G.undirected_edge_name)
    G.add_edge("C", "A", G.directed_edge_name)
    G_copy = G.copy()

    # First test:
    # C -> A -> B
    pc._apply_meek_rule1(G, "A", "B")
    assert G.has_edge("A", "B", G.directed_edge_name)
    assert not G.has_edge("A", "B", G.undirected_edge_name)
    assert G.has_edge("C", "A", G.directed_edge_name)

    # Next, test that nothing occurs
    # if it is a shielded triple
    G = G_copy.copy()
    G.add_edge("C", "B")
    added_arrows = pc._apply_meek_rule1(G, "A", "B")
    assert not added_arrows


def test_pc_rule2():
    edges = [
        ("x", "y"),
        ("z", "y"),
    ]
    G = ADMG(edges)
    oracle = Oracle(G)
    pc = PC(ci_estimator=oracle)

    # If C -> A -> B, with C - B
    # then orient C - B as C -> B
    G = CPDAG()
    G.add_edge("C", "B", G.undirected_edge_name)
    G.add_edge("C", "A", G.directed_edge_name)
    G.add_edge("A", "B", G.directed_edge_name)

    # First test:
    # C -> B
    added_arrows = pc._apply_meek_rule2(G, "C", "B")
    assert G.has_edge("A", "B", G.directed_edge_name)
    assert G.has_edge("C", "A", G.directed_edge_name)
    assert G.has_edge("C", "B", G.directed_edge_name)
    assert added_arrows


def test_pc_rule3():
    G = ADMG()
    oracle = Oracle(G)
    pc = PC(ci_estimator=oracle)

    # If C - A, with C - B and C - D
    # and B -> A <- D, then orient C - A as C -> A.
    G = CPDAG()
    G.add_edge("C", "B", G.undirected_edge_name)
    G.add_edge("C", "D", G.undirected_edge_name)
    G.add_edge("C", "A", G.undirected_edge_name)
    G.add_edge("B", "A", G.directed_edge_name)
    G.add_edge("D", "A", G.directed_edge_name)
    G_copy = G.copy()

    added_arrows = pc._apply_meek_rule3(G, "C", "A")
    assert added_arrows
    assert G.has_edge("C", "A", G.directed_edge_name)

    # if 'B' and 'D' are adjacent, then it will
    # not be added
    G = G_copy.copy()
    G.add_edge("B", "D", G.directed_edge_name)
    added_arrows = pc._apply_meek_rule3(G, "C", "A")
    assert not added_arrows


class Test_PC:
    def setup_method(self):
        # construct a causal graph that will result in
        # x -> y <- z
        G = nx.DiGraph([("x", "y"), ("z", "y")])

        oracle = Oracle(G)

        self.G = G
        self.ci_estimator = oracle
        pc = PC(ci_estimator=self.ci_estimator)
        self.alg = pc

    def test_pc_skel_graph(self):
        sample = dummy_sample(self.G)
        context = make_context().variables(data=sample).build()
        pc = PC(ci_estimator=self.ci_estimator, apply_orientations=False)
        pc.learn_graph(sample, context)
        skel_graph = pc.graph_
        assert all(edge in skel_graph.undirected_edges for edge in {("x", "y"), ("y", "z")})

        for edge in skel_graph.undirected_edges:
            sorted_edge = set(sorted(list(edge)))
            assert sorted_edge in [{"x", "y"}, {"y", "z"}]

    def test_pc_basic_collider(self):
        sample = dummy_sample(self.G)
        context = make_context().variables(data=sample).build()
        pc = PC(ci_estimator=self.ci_estimator, apply_orientations=False)
        pc.learn_graph(sample, context)
        skel_graph = pc.graph_
        sep_set = pc.separating_sets_
        self.alg.orient_unshielded_triples(skel_graph, sep_set)

        # the CPDAG learned should be a fully oriented DAG
        assert len(skel_graph.undirected_edges) == 0
        assert skel_graph.has_edge("x", "y", skel_graph.directed_edge_name)
        assert skel_graph.has_edge("z", "y", skel_graph.directed_edge_name)
