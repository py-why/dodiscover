import networkx as nx
import pywhy_graphs as pgraphs

from dodiscover import InterventionalContextBuilder, make_context
from dodiscover.ci import Oracle
from dodiscover.constraint.skeleton import LearnInterventionSkeleton
from dodiscover.constraint.utils import dummy_sample


def test_fnode_skeleton():
    """Test learning the skeleton for Figure 3 in :footcite:`Kocaoglu2019characterization`."""
    # first create the oracle
    directed_edges = [
        ("x", "w"),
        ("w", "y"),
        ("z", "y"),
    ]
    bidirected_edges = [("x", "z"), ("z", "y")]
    graph = pgraphs.AugmentedGraph(
        incoming_directed_edges=directed_edges, incoming_bidirected_edges=bidirected_edges
    )
    non_f_graph = graph.copy()
    graph.add_f_node({"x"})
    oracle = Oracle(graph)

    # define the expected graph we will learn
    edges = [
        (("F", 0), "x"),
        (("F", 0), "y"),
        ("x", "w"),
        ("x", "z"),
        ("x", "y"),
        ("z", "y"),
        ("w", "y"),
    ]
    expected_skeleton = nx.Graph(edges)
    obs_expected_skeleton = expected_skeleton.copy()
    obs_expected_skeleton.remove_node(("F", 0))

    # define the learner and the context
    learner = LearnInterventionSkeleton(
        ci_estimator=oracle, cd_estimator=oracle, known_intervention_targets=True
    )
    data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
    context = (
        make_context(create_using=InterventionalContextBuilder)
        .variables(data=data[0])
        .intervention_targets([("x",)])
        .build()
    )
    learner.fit(data, context)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph")

    assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)
    assert nx.is_isomorphic(expected_skeleton, skel_graph)


def test_fnode_skeleton_errors():
    pass
