import networkx as nx
import pywhy_graphs as pgraphs

from dodiscover import ContextBuilder, make_context
from dodiscover.cd import KernelCDTest
from dodiscover.ci import FisherZCITest, Oracle
from dodiscover.constraint.skeleton import LearnMultiDomainSkeleton
from dodiscover.constraint.utils import dummy_sample
from dodiscover.datasets import linear


def basic_multidomain_augmented_graph():
    # Create the following graph:
    # F_x -> x -> y -> z
    # S_{1,2} -> y
    # x <--> y
    directed_edges = [
        ("x", "y"),
        ("y", "z"),
    ]
    bidirected_edges = [("x", "y")]
    graph = pgraphs.AugmentedGraph(
        incoming_directed_edges=directed_edges, incoming_bidirected_edges=bidirected_edges
    )
    graph.add_f_node({"x"})
    graph.add_f_node({"x"}, require_unique=False)
    graph.add_s_node((1, 2), {"y"})

    return graph


def test_basic_multidomain_fsnode_skeleton():
    """Test basic skeleton learning with a multidomain f-node and s-node."""
    graph = basic_multidomain_augmented_graph()
    non_f_graph = graph.subgraph(graph.non_augmented_nodes)

    oracle = Oracle(graph, graph.augmented_nodes)

    # define the expected graph we will learn
    edges = [
        (("F", 0), "x"),
        (("F", 0), "y"),
        (("F", 1), "x"),
        (("F", 1), "y"),
        (("S", 0), "y"),
        ("x", "y"),
        ("y", "z"),
    ]
    expected_skeleton = nx.Graph(edges)
    obs_expected_skeleton = expected_skeleton.copy()

    # define the learner and the context
    learner = LearnMultiDomainSkeleton(ci_estimator=oracle, cd_estimator=oracle)
    data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
    domain_indices = [1, 1, 2]
    intervention_targets = [set(), {"x"}, set()]

    context = (
        make_context(create_using=ContextBuilder).variables(data=data[0])
        # .num_distributions(2)
        # .intervention_targets([("x")])
        .build()
    )
    learner.fit(data, context, domain_indices, intervention_targets)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph").subgraph(
        context.observed_variables
    )
    obs_expected_skeleton = obs_expected_skeleton.subgraph(context.observed_variables)
    sep_set = learner.sep_set_

    # check the separating sets
    assert sep_set["x"]["z"] == [{"y", ("F", 0), ("S", 0), ("F", 1)}]

    # check the skeleton after obs data
    print(obs_expected_skeleton.edges())
    print(obs_skel_graph.edges())
    assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)

    # check the skeleton after intervention
    print(skel_graph.edges())
    print(expected_skeleton.edges())
    assert nx.is_isomorphic(expected_skeleton, skel_graph)


def test_basic_multidomain_fsnode_skeleton_with_lindata():
    seed = 1234
    n_samples = 1000
    aug_graph = basic_multidomain_augmented_graph()
    graph = aug_graph.subgraph(aug_graph.non_augmented_nodes)

    # define the expected graph we will learn
    edges = [
        (("F", 0), "x"),
        (("F", 0), "y"),
        (("F", 1), "x"),
        (("F", 1), "y"),
        (("S", 0), "y"),
        ("x", "y"),
        ("y", "z"),
    ]
    expected_skeleton = nx.Graph(edges)
    obs_expected_skeleton = expected_skeleton.copy()

    # define functional relationships of the causal diagram
    graph = pgraphs.functional.make_graph_linear_gaussian(graph, random_state=seed)

    datasets = []
    domain_ids = []
    intervention_sets = []

    # now for each F-node, apply a linear additive intervention
    for f_node, targets in aug_graph.graph["F-nodes"].items():
        new_graph = pgraphs.functional.apply_soft_intervention(
            graph.copy(), targets, random_state=seed
        )

        # generate dataset
        data = linear.sample_from_graph(new_graph, n_samples=n_samples, random_state=seed)

        datasets.append(data)
        intervention_sets.append(targets)
        domain_ids.append(1)

    # now for each S-node, apply a linear additive intervention
    for s_node, targets in aug_graph.graph["S-nodes"].items():
        new_graph = pgraphs.functional.apply_soft_intervention(
            graph.copy(), targets, random_state=seed
        )

        # generate dataset
        data = linear.sample_from_graph(new_graph, n_samples=n_samples, random_state=seed)

        datasets.append(data)
        intervention_sets.append(targets)
        domain_ids.append(2)

    learner = LearnMultiDomainSkeleton(ci_estimator=FisherZCITest(), cd_estimator=KernelCDTest())

    context = make_context(create_using=ContextBuilder).variables(data=datasets[0]).build()
    learner.fit(data, context, domain_ids, intervention_sets)

    # first check the observational skeleton
    skel_graph = learner.adj_graph_
    obs_skel_graph = learner.context_.state_variable("obs_skel_graph").subgraph(
        context.observed_variables
    )
    obs_expected_skeleton = obs_expected_skeleton.subgraph(context.observed_variables)
    sep_set = learner.sep_set_

    # check the separating sets
    assert sep_set["x"]["z"] == [{"y", ("F", 0), ("S", 0), ("F", 1)}]

    # check the skeleton after obs data
    print(obs_expected_skeleton.edges())
    print(obs_skel_graph.edges())
    assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)

    # check the skeleton after intervention
    print(skel_graph.edges())
    print(expected_skeleton.edges())
    assert nx.is_isomorphic(expected_skeleton, skel_graph)
