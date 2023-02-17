from itertools import permutations

import networkx as nx
import numpy as np
import pytest
import pywhy_graphs as pgraphs
from pywhy_graphs import IPAG, PsiPAG

from dodiscover import InterventionalContextBuilder, PsiFCI, make_context
from dodiscover.ci import Oracle
from dodiscover.constraint.utils import dummy_sample

from .test_fcialg import Test_FCI

np.random.seed(12345)


@pytest.mark.filterwarnings("ignore:There is no intervention context set.")
class Test_IFCI(Test_FCI):
    def setup_method(self):
        # construct a causal graph that will result in
        # x -> y <- z
        G = nx.DiGraph([("x", "y"), ("z", "y")])
        oracle = Oracle(G)
        alg = PsiFCI(known_intervention_targets=True, ci_estimator=oracle, cd_estimator=oracle)

        self.context_func = lambda: make_context(create_using=InterventionalContextBuilder)
        self.G = G
        self.ci_estimator = oracle
        self.alg = alg

    def test_rule11(self):
        """Test that all F-nodes are oriented outwards properly."""
        # create a complete graph
        sub_dir_graph = nx.complete_graph(
            [("F", 0), ("F", 1), "a", "b", "c", "d"], create_using=nx.DiGraph
        )
        G = IPAG(incoming_circle_edges=sub_dir_graph)

        # there must only be one kind of edge from F-nodes to its nbrs
        f_nodes = [("F", 0), ("F", 1)]
        self.alg._apply_rule11(G, f_nodes)
        for f_node in f_nodes:
            for nbr in G.neighbors(f_node):
                if nbr in f_nodes:
                    continue
                assert G.has_edge(f_node, nbr, G.directed_edge_name)
                assert not G.has_edge(nbr, f_node)

    def test_rule12(self):
        """Test rule "9" in the I-FCI paper from Figure 3."""
        # create a complete graph
        directed_edges = [
            (("F", 0), "x"),
            (("F", 0), "y"),
            ("z", "x"),
            ("z", "y"),
            ("x", "y"),
            ("x", "w"),
            ("w", "y"),
        ]
        circle_edges = [("y", "x"), ("x", "z"), ("y", "z"), ("y", "w")]
        G = IPAG(incoming_directed_edges=directed_edges, incoming_circle_edges=circle_edges)
        G.graph["F-nodes"][("F", 0)] = ["x"]
        f_nodes = G.f_nodes

        # there must only be one kind of edge from F-nodes to its nbrs
        symmetric_diff_map = {
            ("F", 0): ["x"],
        }

        # no arrows should be added if we are not operating over a F-node
        for x, y, z in permutations(G.non_f_nodes, 3):
            added_arrows = self.alg._apply_rule12(G, x, y, z, f_nodes, symmetric_diff_map)
            assert not added_arrows

        # no arrows should be added if the conditions of the rule are not met
        added_arrows = self.alg._apply_rule12(G, ("F", 0), "x", "z", f_nodes, symmetric_diff_map)
        assert not added_arrows

        added_arrows = self.alg._apply_rule12(G, ("F", 0), "x", "y", f_nodes, symmetric_diff_map)
        if self.alg.known_intervention_targets:
            assert added_arrows
            assert G.has_edge("x", "y", G.directed_edge_name)
            assert not G.has_edge("y", "x")
        else:
            assert not added_arrows
            assert G.has_edge("y", "x")

    def test_ifci_figure3(self):
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
        learner = PsiFCI(ci_estimator=oracle, cd_estimator=oracle, known_intervention_targets=True)
        data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
        context = (
            make_context(create_using=InterventionalContextBuilder)
            .variables(data=data[0])
            .intervention_targets([("x",)])
            .build()
        )
        learner.fit(data, context)

        # first check the observational skeleton
        skel_graph = learner.skeleton_learner_.adj_graph_
        obs_skel_graph = learner.skeleton_learner_.context_.state_variable("obs_skel_graph")

        assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)
        assert nx.is_isomorphic(expected_skeleton, skel_graph)

        # now check the end graph
        directed_edges = [
            (("F", 0), "x"),
            (("F", 0), "y"),
            ("z", "x"),
            ("z", "y"),
            ("x", "y"),
            ("x", "w"),
            ("w", "y"),
        ]
        circle_edges = [("x", "z"), ("y", "z"), ("y", "w")]
        expected_G = IPAG(
            incoming_directed_edges=directed_edges, incoming_circle_edges=circle_edges
        )
        expected_G.graph["F-nodes"][("F", 0)] = ["x"]

        learned_graph = learner.graph_
        for edge_type, subgraph in expected_G.get_graphs().items():
            assert nx.is_isomorphic(subgraph, learned_graph.get_graphs(edge_type))


@pytest.mark.filterwarnings("ignore:There is no intervention context set.")
class Test_PsiFCI(Test_IFCI):
    def setup_method(self):
        # construct a causal graph that will result in
        # x -> y <- z
        G = nx.DiGraph([("x", "y"), ("z", "y")])
        oracle = Oracle(G)
        alg = PsiFCI(known_intervention_targets=False, ci_estimator=oracle, cd_estimator=oracle)

        self.context_func = lambda: make_context(create_using=InterventionalContextBuilder)
        self.G = G
        self.ci_estimator = oracle
        self.alg = alg

    def test_figure2_skeleton(self):
        """Test learning the graph for Figure 2 in :footcite:`Jaber2020causal`."""
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
        learner = PsiFCI(ci_estimator=oracle, cd_estimator=oracle, known_intervention_targets=False)
        data = [dummy_sample(non_f_graph), dummy_sample(non_f_graph)]
        context = (
            make_context(create_using=InterventionalContextBuilder)
            .variables(data=data[0])
            .num_distributions(2)
            .build()
        )
        learner.fit(data, context)

        # first check the observational skeleton
        skel_graph = learner.skeleton_learner_.adj_graph_
        obs_skel_graph = learner.skeleton_learner_.context_.state_variable("obs_skel_graph")

        assert nx.is_isomorphic(obs_expected_skeleton, obs_skel_graph, edge_match=None)
        assert nx.is_isomorphic(expected_skeleton, skel_graph)

        # now check the end graph
        directed_edges = [
            (("F", 0), "x"),
            (("F", 0), "w"),
            ("z", "x"),
            ("z", "y"),
            ("z", "w"),
            ("x", "y"),
            ("x", "w"),
            ("y", "w"),
        ]
        circle_edges = [("x", "z"), ("w", "x"), ("w", "z"), ("w", "y")]
        expected_G = PsiPAG(
            incoming_directed_edges=directed_edges, incoming_circle_edges=circle_edges
        )
        expected_G.graph["F-nodes"][("F", 0)] = ["x"]

        learned_graph = learner.graph_
        for edge_type, subgraph in expected_G.get_graphs().items():
            assert nx.is_isomorphic(subgraph, learned_graph.get_graphs(edge_type))
