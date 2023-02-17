import networkx as nx
import pytest
import pywhy_graphs as pgraphs

from dodiscover import InterventionalContextBuilder, make_context
from dodiscover.ci import Oracle
from dodiscover.constraint.skeleton import LearnInterventionSkeleton
from dodiscover.constraint.utils import dummy_sample


def test_fnode_skeleton_known_targets():
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


def test_fnode_skeleton_unknown_targets():
    """Test learning the skeleton for Figure 2 in :footcite:`Jaber2020causal`."""
    # first create the oracle
    directed_edges = [
        ("x", "w"),
        ("x", "y"),
        ("y", "w"),
        ("z", "y"),
        ("z", "x"),
    ]
    bidirected_edges = [("x", "w")]
    graph = pgraphs.AugmentedGraph(
        incoming_directed_edges=directed_edges, incoming_bidirected_edges=bidirected_edges
    )
    non_f_graph = graph.copy()
    graph.add_f_node({"x", "w"})
    oracle = Oracle(graph)

    # define the expected graph we will learn
    edges = [
        (("F", 0), "x"),
        (("F", 0), "w"),
        ("x", "w"),
        ("x", "z"),
        ("x", "y"),
        ("z", "y"),
        ("z", "w"),
        ("w", "y"),
    ]
    expected_skeleton = nx.Graph(edges)
    obs_expected_skeleton = expected_skeleton.copy()
    obs_expected_skeleton.remove_node(("F", 0))

    # define the learner and the context
    learner = LearnInterventionSkeleton(
        ci_estimator=oracle, cd_estimator=oracle, known_intervention_targets=False
    )
    data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
    context = (
        make_context(create_using=InterventionalContextBuilder)
        .variables(data=data[0])
        .num_distributions(2)
        .build()
    )
    learner.fit(data, context)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph")

    assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)
    assert nx.is_isomorphic(expected_skeleton, skel_graph)


def test_fnode_skeleton_errors():
    # define the learner and the context
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

    data = [dummy_sample(non_f_graph)]
    learner = LearnInterventionSkeleton(
        ci_estimator=oracle, cd_estimator=oracle, known_intervention_targets=True
    )
    context = (
        make_context(create_using=InterventionalContextBuilder)
        .variables(data=data[0])
        .intervention_targets([("x",)])
        .build()
    )

    with pytest.raises(RuntimeError, match="The number of datasets does not match"):
        learner.fit(data, context)
