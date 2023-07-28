import networkx as nx
import pytest
import pywhy_graphs as pgraphs

from dodiscover import Context, InterventionalContextBuilder, make_context
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
    learner.learn_graph(data, context)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph").subgraph(
        context.get_non_augmented_nodes()
    )

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

    # import pywhy_graphs.networkx as pywhy_nx
    # # 0 y ('F', 0) {'x'}
    # print(pywhy_nx.m_separated(graph, {'y'}, ('F', 0), {'x'}))
    # print(oracle.test(dummy_sample(graph), {'y'}, {('F', 0)}, {'x'}))

    # define the learner and the context
    learner = LearnInterventionSkeleton(
        ci_estimator=oracle, cd_estimator=oracle, known_intervention_targets=False
    )
    data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
    context: Context = (
        make_context(create_using=InterventionalContextBuilder)
        .variables(data=data[0])
        .num_distributions(2)
        .build()
    )
    learner.learn_graph(data, context)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph").subgraph(
        context.get_non_augmented_nodes()
    )

    print(obs_expected_skeleton.edges())
    print(obs_skel_graph.edges())
    assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)
    print(expected_skeleton.edges())
    print(skel_graph.edges())
    for edge in skel_graph.edges():
        if not expected_skeleton.has_edge(*edge):
            print(edge)
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
        learner.learn_graph(data, context)


def test_basic_fnode_skeleton():
    """Test the F-nodes are part of the separating set."""
    directed_edges = [
        ("x", "y"),
        ("y", "z"),
    ]
    bidirected_edges = [("x", "y")]
    graph = pgraphs.AugmentedGraph(
        incoming_directed_edges=directed_edges, incoming_bidirected_edges=bidirected_edges
    )
    non_f_graph = graph.copy()
    graph.add_f_node({"x", "z"})
    oracle = Oracle(graph, graph.f_nodes)

    # define the expected graph we will learn
    edges = [
        (("F", 0), "x"),
        (("F", 0), "y"),
        (("F", 0), "z"),
        ("x", "y"),
        ("y", "z"),
    ]
    expected_skeleton = nx.Graph(edges)
    obs_expected_skeleton = expected_skeleton.copy()
    obs_expected_skeleton.remove_node(("F", 0))

    # define the learner and the context
    learner = LearnInterventionSkeleton(
        ci_estimator=oracle,
        cd_estimator=oracle,
        known_intervention_targets=True,
    )
    data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
    context: Context = (
        make_context(create_using=InterventionalContextBuilder)
        .variables(data=data[0])
        .num_distributions(2)
        .intervention_targets([("x", "z")])
        .build()
    )
    learner.learn_graph(data, context)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph").subgraph(
        context.get_non_augmented_nodes()
    )
    sep_set = learner.sep_set_

    # check the separating sets
    # XXX: CAN IMPROVE THE ASSERTION if we can get separating sets to only be checked once..
    assert {"y", ("F", 0)} in sep_set["x"]["z"]

    # check the skeleton after obs data
    print(obs_expected_skeleton.edges())
    print(obs_skel_graph.edges())
    assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)

    # check the skeleton after intervention
    print(skel_graph.edges())
    print(expected_skeleton.edges())
    assert nx.is_isomorphic(expected_skeleton, skel_graph)
