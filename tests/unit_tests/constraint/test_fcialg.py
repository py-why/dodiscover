import logging

import networkx as nx
import numpy as np
import pytest
import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs import ADMG, PAG
from pywhy_graphs.testing import assert_mixed_edge_graphs_isomorphic

from dodiscover import FCI, make_context
from dodiscover.ci import Oracle
from dodiscover.constraint.config import ConditioningSetSelection
from dodiscover.constraint.utils import dummy_sample

np.random.seed(12345)


class Test_FCI:
    def setup_method(self):
        # construct a causal graph that will result in
        # x -> y <- z
        G = nx.DiGraph([("x", "y"), ("z", "y")])
        oracle = Oracle(G)
        fci = FCI(ci_estimator=oracle)

        self.context_func = make_context
        self.G = G
        self.ci_estimator = oracle
        self.alg = fci

    def test_fci_skel_graph(self):
        sample = dummy_sample(self.G)
        context = self.context_func().variables(data=sample).build()
        skel_graph, _ = self.alg.learn_skeleton(sample, context)
        assert nx.is_isomorphic(skel_graph, self.G.to_undirected())

    def test_fci_basic_collider(self):
        sample = dummy_sample(self.G)
        context = self.context_func().variables(data=sample).build()
        skel_graph, sep_set = self.alg.learn_skeleton(sample, context)
        graph = PAG(incoming_circle_edges=skel_graph)
        self.alg.orient_unshielded_triples(graph, sep_set)

        # the PAG learned x o-> y <-o z
        expected_graph = PAG()
        expected_graph.add_edges_from([("x", "y"), ("z", "y")], expected_graph.directed_edge_name)
        expected_graph.add_edges_from([("y", "x"), ("y", "z")], expected_graph.circle_edge_name)
        assert set(expected_graph.edges()[expected_graph.directed_edge_name]) == set(
            graph.edges()[expected_graph.directed_edge_name]
        )
        assert set(expected_graph.edges()[expected_graph.circle_edge_name]) == set(
            graph.edges()[expected_graph.circle_edge_name]
        )

    def test_fci_rule1(self):
        # If A *-> u o-* C, A and C are not adjacent,
        # then we can orient the triple as A *-> u -> C.

        # First test:
        # A -> u o-o C
        G = PAG()
        G.add_edge("A", "u", G.directed_edge_name)
        G.add_edge("u", "C", G.circle_edge_name)
        G.add_edge("C", "u", G.circle_edge_name)
        G_copy = G.copy()

        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C", G.directed_edge_name)
        assert not G.has_edge("C", "u", G.circle_edge_name)
        assert not G.has_edge("C", "u", G.directed_edge_name)
        assert not G.has_edge("u", "A", G.directed_edge_name)

        # orient u o-o C now as u o-> C
        # Second test:
        # A -> u o-> C
        G = G_copy.copy()
        G.orient_uncertain_edge("u", "C")
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C", G.directed_edge_name)
        assert not G.has_edge("C", "u", G.circle_edge_name)
        assert not G.has_edge("C", "u", G.directed_edge_name)
        assert not G.has_edge("u", "A", G.directed_edge_name)

        # now orient A -> u as A <-> u
        # Third test:
        # A <-> u o-o C
        G = G_copy.copy()
        G.remove_edge("A", "u", G.directed_edge_name)
        G.add_edge("u", "A", G.bidirected_edge_name)
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C", G.directed_edge_name)
        assert not G.has_edge("C", "u", G.circle_edge_name)
        assert not G.has_edge("C", "u", G.directed_edge_name)
        assert G.has_edge("u", "A", G.bidirected_edge_name)

        # now orient A -> u as A <-> u
        # Fourth test:
        # A o-> u o-o C
        G = G_copy.copy()
        G.add_edge("u", "A", G.circle_edge_name)
        G.orient_uncertain_edge("u", "C")
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C", G.directed_edge_name)
        assert not G.has_edge("C", "u", G.circle_edge_name)
        assert not G.has_edge("C", "u", G.directed_edge_name)
        assert G.has_edge("u", "A", G.circle_edge_name)

        # check that no orientation happens if A, C are adjacent
        G = G_copy.copy()
        G.add_edge("A", "C", G.directed_edge_name)
        added_arrows = self.alg._apply_rule1(G, "u", "A", "C")
        assert not added_arrows

    def test_fci_rule2(self):
        # If A -> u *-> C, or A *-> u -> C, and A *-o C, then
        # orient A *-> C.
        # 1. Do A -> u <-> C with A o-o C
        G = PAG()
        G.add_edge("A", "u", G.directed_edge_name)
        G.add_edge("u", "C", G.bidirected_edge_name)
        G.add_edge("A", "C", G.circle_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        G_copy = G.copy()

        self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_edge("A", "C", G.directed_edge_name)
        assert G.has_edge("C", "A", G.circle_edge_name)

        # if A o-> u, then it should not work
        G = G_copy.copy()
        G.add_edge("u", "A", G.circle_edge_name)
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert not added_arrows
        assert G.has_edge("A", "C", G.circle_edge_name)
        assert G.has_edge("C", "A", G.circle_edge_name)

        # 2. Test not-added case
        # first test that can't be A <-> u <-> C
        G = G_copy.copy()
        G.remove_edge("A", "u", G.directed_edge_name)
        G.add_edge("u", "A", G.bidirected_edge_name)
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_edge("A", "C", G.circle_edge_name)
        assert not added_arrows

        # 3. then test that A <-> u -> C with A o-o C
        G.remove_edge("C", "u", G.bidirected_edge_name)
        G.add_edge("u", "C", G.directed_edge_name)
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_edge("A", "C", G.directed_edge_name)
        assert G.has_edge("C", "A", G.circle_edge_name)
        assert added_arrows

    def test_fci_rule3(self):
        # If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
        # and v *-o u, then orient v *-> u.
        G = PAG()

        # start by considering all stars to be empty for A, C, u
        G.add_edge("A", "u", G.directed_edge_name)
        G.add_edge("C", "u", G.directed_edge_name)

        # then consider all circles as bidirected
        G.add_edge("A", "v", G.circle_edge_name)
        G.add_edge("v", "A", G.circle_edge_name)
        G.add_edge("C", "v", G.circle_edge_name)
        G.add_edge("v", "C", G.circle_edge_name)
        G.add_edge("v", "u", G.circle_edge_name)
        G.add_edge("u", "v", G.circle_edge_name)
        G_copy = G.copy()

        self.alg._apply_rule3(G, "u", "A", "C")
        for edge in G_copy.edges()[G.directed_edge_name]:
            assert G.has_edge(*edge, G.directed_edge_name)
        for edge in G_copy.edges()[G.circle_edge_name]:
            if edge != ("v", "u"):
                assert G.has_edge(*edge, G.circle_edge_name)
            else:
                assert not G.has_edge(*edge, G.circle_edge_name)
        assert G.has_edge("v", "u", G.directed_edge_name)

        # if A -> u is A <-> u, then it should still work
        G = G_copy.copy()
        G.remove_edge("A", "u", G.directed_edge_name)
        G.add_edge("A", "u", G.bidirected_edge_name)
        added_arrows = self.alg._apply_rule3(G, "u", "A", "C")
        assert added_arrows

        # adding a circle edge should make it not work
        G = G_copy.copy()
        G.add_edge("A", "C", G.circle_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        added_arrows = self.alg._apply_rule3(G, "u", "A", "C")
        assert not added_arrows

    def test_fci_rule4_without_sepset(self):
        """Test orienting a discriminating path without separating set.

        A discriminating path, p, between X and Y is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        <v,..., w, u, c>
        """
        G = PAG()

        # setup graph with a <-> u o-o c
        G.add_edge("u", "c", G.circle_edge_name)
        G.add_edge("c", "u", G.circle_edge_name)
        G.add_edge("a", "u", G.bidirected_edge_name)
        sep_set = set()

        # initial test should not add any arrows, since there are only 2 edges
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == set()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        # setup graph with b <-> a <-> u o-o c
        G.add_edge("b", "a", G.bidirected_edge_name)
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == set()

        # add the arrow from a -> c
        G.add_edge("a", "c", G.directed_edge_name)
        G_copy = G.copy()
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert explored_nodes == set(["c", "u", "a", "b"])

        # since separating set is empty
        assert not G.has_edge("c", "u", G.circle_edge_name)
        assert G.has_edge("c", "u", G.bidirected_edge_name)

        # change 'u' o-o 'c' to 'u' o-> 'c', which should now orient
        # the same way
        G = G_copy.copy()
        G.orient_uncertain_edge("u", "c")
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert explored_nodes == set(["c", "u", "a", "b"])
        assert not G.has_edge("c", "u", G.circle_edge_name)
        assert G.has_edge("c", "u", G.bidirected_edge_name)

    def test_fci_rule4_early_exit(self, caplog):
        G = PAG()

        G.add_edge("u", "c", G.circle_edge_name)
        G.add_edge("c", "u", G.circle_edge_name)
        G.add_edge("a", "u", G.bidirected_edge_name)
        sep_set = set()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        G.add_edge("b", "a", G.bidirected_edge_name)
        G.add_edge("a", "c", G.directed_edge_name)
        G.add_edge("b", "c", G.directed_edge_name)
        G.add_edge("d", "b", G.directed_edge_name)

        # test error case
        with caplog.at_level(logging.WARNING):
            new_fci = FCI(ci_estimator=self.ci_estimator, max_path_length=1)
            new_fci._apply_rule4(G, "u", "a", "c", sep_set)
            assert "Did not finish checking" in caplog.text

    def test_fci_rule4_wit_sepset(self):
        """Test orienting a discriminating path with a separating set.

        A discriminating path, p, between X and Y is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        <v,..., w, u, c>
        """
        G = PAG()

        G.add_edge("u", "c", G.circle_edge_name)
        G.add_edge("c", "u", G.circle_edge_name)
        G.add_edge("a", "u", G.bidirected_edge_name)
        sep_set = {"b": {"c": set("u")}}

        # initial test should not add any arrows, since there are only 2 edges
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == set()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        G.add_edge("b", "a", G.bidirected_edge_name)
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == set()

        # add the arrow from a -> c
        G.add_edge("a", "c", G.directed_edge_name)
        G_copy = G.copy()
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert explored_nodes == set(["c", "u", "a", "b"])
        assert not G.has_edge("c", "u", G.circle_edge_name)
        assert not G.has_edge("c", "u", G.directed_edge_name)
        assert G.has_edge("u", "c", G.directed_edge_name)

        # change 'u' o-o 'c' to 'u' o-> 'c', which should now orient
        # the same way
        G = G_copy.copy()
        G.orient_uncertain_edge("u", "c")
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert explored_nodes == set(["c", "u", "a", "b"])
        assert not G.has_edge("c", "u", G.circle_edge_name)
        assert not G.has_edge("c", "u", G.directed_edge_name)
        assert G.has_edge("u", "c", G.directed_edge_name)

    def test_fci_rule5(self):
        # Let A o-o B, and uncovered circle path A o-o G o-o M o-o T o-o B where A, T are
        # not adjacent, and B, G are not adjacent. Then A - G - M - T - B - A.
        G = PAG()
        circled_edges = [("A", "G"), ("G", "M"), ("M", "T"), ("T", "B"), ("A", "B")]
        for a, b in circled_edges:
            G.add_edge(a, b, G.circle_edge_name)
            G.add_edge(b, a, G.circle_edge_name)

        added_tails = self.alg._apply_rule5(G, "B", "A")

        assert added_tails
        for a, b in circled_edges:
            assert G.has_edge(a, b, G.undirected_edge_name)
            assert not G.has_edge(a, b, G.circle_edge_name)
            assert not G.has_edge(b, a, G.circle_edge_name)

    def test_fci_rule6(self):
        # If A - B o-* C then A - B -* C

        # Check for directed edge: if A - B o-> C then A - B -> C
        G = PAG()
        G.add_edge("A", "B", G.undirected_edge_name)
        G.add_edge("C", "B", G.circle_edge_name)
        G.add_edge("B", "C", G.directed_edge_name)

        added_tails = self.alg._apply_rule6(G, "B", "A", "C")

        assert G.has_edge("B", "C", G.directed_edge_name)
        assert not G.has_edge("C", "B", G.circle_edge_name)
        # Check for birected edge: if A - B o- C then A - B - C
        G = PAG()
        G.add_edge("A", "B", G.undirected_edge_name)
        G.add_edge("C", "B", G.circle_edge_name)

        added_tails = self.alg._apply_rule6(G, "B", "A", "C")
        assert added_tails

        assert G.has_edge("B", "C", G.undirected_edge_name)
        assert not G.has_edge("C", "B", G.circle_edge_name)

        # Check for birected edge: if A - B o-o C then A - B -o C
        G = PAG()
        G.add_edge("A", "B", G.undirected_edge_name)
        G.add_edge("C", "B", G.circle_edge_name)
        G.add_edge("B", "C", G.circle_edge_name)

        added_tails = self.alg._apply_rule6(G, "B", "A", "C")

        assert added_tails
        assert G.has_edge("B", "C", G.circle_edge_name)
        assert not G.has_edge("C", "B", G.circle_edge_name)

    def test_fci_rule7(self):
        # Check for directed edge: if A -o B o-> C then A -o B -> C
        G = PAG()
        G.add_edge("A", "U", G.circle_edge_name)
        G.add_edge("U", "C", G.directed_edge_name)
        G.add_edge("C", "U", G.circle_edge_name)
        G_copy = G.copy()

        added_tails = self.alg._apply_rule7(G, "U", "A", "C")

        assert added_tails
        assert not G.has_edge("C", "U", G.circle_edge_name)

        # Check for directed edge: if A -o B o-> C but A -> C then nothing happens
        G = G_copy.copy()
        G.add_edge("A", "C", G.directed_edge_name)
        added_tails = self.alg._apply_rule7(G, "U", "A", "C")
        assert not added_tails
        assert G.has_edge("C", "U", G.circle_edge_name)

    def test_fci_rule8_without_selection_bias(self):
        # If A -> u -> C and A o-> C
        # orient A o-> C as A -> C
        G = PAG()

        # create a chain for A, u, C
        G.add_edges_from([("A", "u"), ("u", "C")], G.directed_edge_name)
        G.add_edge("A", "C", G.directed_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        self.alg._apply_rule8(G, "u", "A", "C")

        assert G.has_edge("A", "C", G.directed_edge_name)
        assert not G.has_edge("C", "A", G.circle_edge_name)

        # Check that if A o-> u -> C and A o-> C then rule is not applied
        G = PAG()
        G.add_edge("A", "u", G.directed_edge_name)
        G.add_edge("u", "C", G.directed_edge_name)
        G.add_edge("u", "A", G.circle_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        G.add_edge("A", "C", G.directed_edge_name)

        assert not self.alg._apply_rule8(G, "u", "A", "C")

        # Check that if A <-o u -> C and A o-> C then rule is not applied
        G = PAG()
        G.add_edge("A", "u", G.circle_edge_name)
        G.add_edge("u", "C", G.directed_edge_name)
        G.add_edge("u", "A", G.directed_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        G.add_edge("A", "C", G.directed_edge_name)

        assert not self.alg._apply_rule8(G, "u", "A", "C")

    def test_fci_rule8_with_selection_bias(self):
        if not self.alg.selection_bias:
            pytest.skip(reason="No selection bias for this algorithm")

        # If A -o u -> C and A o-> C then orient A o-> C as A -> C
        G = PAG()

        # create a chain for A, u, C
        G.add_edge("u", "C", G.directed_edge_name)
        G.add_edge("A", "u", G.circle_edge_name)
        G.add_edge("A", "C", G.directed_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        self.alg._apply_rule8(G, "u", "A", "C")

        assert G.has_edge("A", "C", G.directed_edge_name)
        assert not G.has_edge("C", "A", G.circle_edge_name)

    def test_fci_rule9(self):
        # If A o-> C and there is an undirected pd path
        # from A to C through u, where u and C are not adjacent
        # then orient A o-> C as A -> C
        G = PAG()

        # create an uncovered pd path from A to C through u
        G.add_edge("A", "C", G.directed_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        G.add_edges_from(
            [("A", "u"), ("u", "x"), ("x", "y"), ("y", "z"), ("z", "C")], G.directed_edge_name
        )
        G.add_edge("y", "x", G.circle_edge_name)

        # create a pd path from A to C through v
        G.add_edges_from(
            [("A", "v"), ("v", "x"), ("x", "y"), ("y", "z"), ("z", "C")], G.directed_edge_name
        )
        # with the bidirected edge, v,x,y is a shielded triple
        G.add_edge("v", "y", G.bidirected_edge_name)
        G_copy = G.copy()

        # get the uncovered pd paths
        added_arrows, uncov_pd_path = self.alg._apply_rule9(G, "u", "A", "C")
        assert added_arrows
        assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]
        assert not G.has_edge("C", "A", G.circle_edge_name)

        # the shielded triple should not result in an uncovered pd path
        G = G_copy.copy()
        added_arrows, uncov_pd_path = self.alg._apply_rule9(G, "v", "A", "C")
        assert not added_arrows
        assert uncov_pd_path == []
        assert G.has_edge("C", "A", G.circle_edge_name)

        # when there is a circle edge it should still work
        G = G_copy.copy()
        G.add_edge("C", "z", G.circle_edge_name)
        added_arrows, uncov_pd_path = self.alg._apply_rule9(G, "u", "A", "C")
        assert added_arrows
        assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]
        assert not G.has_edge("C", "A", G.circle_edge_name)

        # test fig 6 Zhang 2008
        G = PAG()
        G.add_edges_from(
            [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("D", "B"), ("D", "C")],
            G.circle_edge_name,
        )
        G.add_edges_from([("B", "D"), ("C", "D")], G.directed_edge_name)
        self.alg._apply_rule9(G, "A", "B", "D")

        expected_G = PAG()
        expected_G.add_edges_from(
            [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("D", "C")],
            expected_G.circle_edge_name,
        )
        expected_G.add_edges_from([("B", "D"), ("C", "D")], expected_G.directed_edge_name)

        assert G.edges() == expected_G.edges()
        self.alg._apply_rule9(G, "A", "C", "D")
        expected_G.remove_edge("D", "C", expected_G.circle_edge_name)

        assert G.edges() == expected_G.edges()

    def test_fci_rule10(self):
        # If A o-> C and u -> C <- v and:
        # - there is an uncovered pd path from A to u, p1
        # - there is an uncovered pd from from A to v, p2
        # if mu adjacent to A on p1 is distinct from w adjacent to A on p2
        # and mu is not adjacent to w, then orient orient A o-> C as A -> C
        G = PAG()

        # make A o-> C
        G.add_edge("A", "C", G.directed_edge_name)
        G.add_edge("C", "A", G.circle_edge_name)
        # create an uncovered pd path from A to u that ends at C
        G.add_edges_from(
            [("A", "x"), ("x", "y"), ("y", "z"), ("z", "u"), ("u", "C")], G.directed_edge_name
        )
        G.add_edge("y", "x", G.circle_edge_name)

        # create an uncovered pd path from A to v so now C is a collider for <u, C, v>
        G.add_edge("z", "v", G.directed_edge_name)
        G.add_edge("v", "C", G.directed_edge_name)
        G_copy = G.copy()

        # 'x' and 'x' are not distinct, so won't orient
        added_arrows, a_to_u_path, a_to_v_path = self.alg._apply_rule10(G, "u", "A", "C")
        assert not added_arrows
        assert a_to_u_path == []
        assert a_to_v_path == []
        assert G.has_edge("C", "A", G.circle_edge_name)

        # if we create an edge from A -> y, there is now a distinction
        G = G_copy.copy()
        G.add_edges_from([("A", "xy"), ("xy", "y")], G.directed_edge_name)
        added_arrows, a_to_u_path, a_to_v_path = self.alg._apply_rule10(G, "u", "A", "C")
        assert added_arrows
        assert a_to_u_path in (["A", "x", "y", "z", "u"], ["A", "xy", "y", "z", "u"])
        assert a_to_v_path in (["A", "xy", "y", "z", "v"], ["A", "x", "y", "z", "v"])

        # by making one edge not potentially directed, we break R10
        G.remove_edge("z", "u", G.directed_edge_name)
        G.add_edge("u", "z", G.directed_edge_name)
        added_arrows, a_to_u_path, a_to_v_path = self.alg._apply_rule10(G, "u", "A", "C")
        assert not added_arrows
        assert a_to_u_path == []
        assert a_to_v_path == []
        G.add_edge("z", "u", G.circle_edge_name)
        added_arrows, a_to_u_path, a_to_v_path = self.alg._apply_rule10(G, "u", "A", "C")
        assert not added_arrows
        assert a_to_u_path == []
        assert a_to_v_path == []

    def test_fci_unobserved_confounder(self):
        # x4 -> x2 <- x1 <- x3
        # x1 <--> x2
        # x4 | x1,
        edge_list = [
            ("x4", "x2"),
            ("x3", "x1"),
            ("x1", "x2"),
        ]
        latent_edge_list = [("x1", "x2")]
        G = ADMG(edge_list, incoming_bidirected_edges=latent_edge_list)
        sample = dummy_sample(G)
        context = self.context_func().variables(data=sample).build()

        oracle = Oracle(G)
        ci_estimator = oracle
        fci = FCI(ci_estimator=ci_estimator)
        fci.learn_graph(sample, context)
        pag = fci.graph_

        expected_pag = PAG()
        expected_pag.add_edges_from(
            [
                ("x4", "x2"),
                ("x1", "x2"),
                ("x3", "x2"),
            ],
            expected_pag.directed_edge_name,
        )
        expected_pag.add_edges_from(
            [("x2", "x4"), ("x2", "x3"), ("x2", "x1"), ("x1", "x3"), ("x3", "x1")],
            expected_pag.circle_edge_name,
        )

        assert set(pag.edges()) == set(expected_pag.edges())

    def test_fci_spirtes_example(self):
        """Test example in book.

        See book Figure 16

        See: https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/.g/scottd/fullbook.pdf
        """  # noqa: E501
        # reconstruct the PAG the way FCI would
        edge_list = [("D", "A"), ("B", "E"), ("F", "B"), ("C", "F"), ("C", "H"), ("H", "D")]
        latent_edge_list = [("A", "B"), ("D", "E")]
        graph = ADMG(edge_list, latent_edge_list)
        alg = FCI(ci_estimator=Oracle(graph))
        sample = dummy_sample(graph)
        context = self.context_func().variables(data=sample).build()
        alg.learn_graph(sample, context)
        pag = alg.graph_

        # generate the expected PAG
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
        expected_pag = PAG(
            edge_list,
            incoming_bidirected_edges=latent_edge_list,
            incoming_circle_edges=uncertain_edge_list,
        )
        assert_mixed_edge_graphs_isomorphic(pag, expected_pag)

    @pytest.mark.parametrize(
        "condsel_method",
        [
            ConditioningSetSelection.NBRS,
            ConditioningSetSelection.NBRS_PATH,
            ConditioningSetSelection.COMPLETE,
        ],
    )
    @pytest.mark.parametrize(
        "pds_condsel_method", [ConditioningSetSelection.PDS, ConditioningSetSelection.PDS_PATH]
    )
    @pytest.mark.parametrize("selection_bias", [True, False])
    def test_fci_complex(self, condsel_method, pds_condsel_method, selection_bias):
        """
        Test FCI algorithm with more complex graph.

        Use Figure 2 from :footcite:`Colombo2012`.

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
        latent_edge_list = [("x1", "x2"), ("x4", "x5")]
        G = ADMG(edge_list, latent_edge_list)
        sample = dummy_sample(G)
        context = self.context_func().variables(data=sample).build()
        oracle = Oracle(G)
        ci_estimator = oracle
        fci = FCI(
            ci_estimator=ci_estimator,
            max_iter=np.inf,
            condsel_method=condsel_method,
            pds_condsel_method=pds_condsel_method,
            selection_bias=selection_bias,
        )
        fci.learn_graph(sample, context)
        pag = fci.graph_

        # double check the m-separation statement and PDS
        assert pywhy_nx.m_separated(G, {"x1"}, {"x3"}, {"x4"})
        pdsep = pywhy_graphs.pds(G, "x1", "x3")
        assert "x2" in pdsep

        expected_pag = PAG()
        expected_pag.add_edges_from(
            [("x6", "x5"), ("x2", "x3"), ("x4", "x3"), ("x6", "x4")], expected_pag.circle_edge_name
        )
        expected_pag.add_edges_from(
            [
                ("x4", "x1"),
                ("x2", "x5"),
                ("x3", "x2"),
                ("x3", "x4"),
                ("x2", "x6"),
                ("x3", "x6"),
                ("x4", "x6"),
                ("x5", "x6"),
            ],
            expected_pag.directed_edge_name,
        )
        expected_pag.add_edge("x1", "x2", expected_pag.bidirected_edge_name)
        expected_pag.add_edge("x4", "x5", expected_pag.bidirected_edge_name)

        assert set(pag.edges()) == set(expected_pag.edges())
        assert_mixed_edge_graphs_isomorphic(pag, expected_pag)

    def test_fci_fig6(self):
        """
        Based on Figure 6 from :footcite:`Zhang2008`

        """

        import pywhy_graphs

        # Pretend this is a MAG - refactor if MAGs are developed
        G = ADMG()
        G.add_edge("A", "C", G.directed_edge_name)
        G.add_edge("A", "B", G.bidirected_edge_name)
        G.add_edge("B", "D", G.directed_edge_name)
        G.add_edge("C", "D", G.directed_edge_name)
        assert pywhy_graphs.networkx.m_separated(G, {"A"}, {"D"}, {"B", "C"})

        sample = dummy_sample(G)
        context = self.context_func().variables(data=sample).build()
        oracle = Oracle(G)
        ci_estimator = oracle
        fci = FCI(ci_estimator=ci_estimator, max_iter=np.inf, selection_bias=False)
        fci.learn_graph(sample, context)
        pag = fci.graph_

        expected_G = PAG()
        expected_G.add_edge("A", "B", expected_G.circle_edge_name)
        expected_G.add_edge("B", "A", expected_G.circle_edge_name)
        expected_G.add_edge("C", "A", expected_G.circle_edge_name)
        expected_G.add_edge("A", "C", expected_G.circle_edge_name)
        expected_G.add_edge("B", "D", expected_G.directed_edge_name)
        expected_G.add_edge("C", "D", expected_G.directed_edge_name)

        assert pag.nodes() == expected_G.nodes()
        assert pag.edges() == expected_G.edges()

    def test_fci_selection_bias(self):
        """
        Based on Figure 1 from :footcite:`Zhang2008`, with extra edge R <- D

        The DAG (over observed and selected variables) is
        A -> Ef <-> R <- D, Ef -> Sel, where Sel is a selection variable.

        The MAG is A - Ef -> R <- D.

        The PAG is A o-o Ef o-o R o-o D.

        References
        ----------
        .. footbibliography::

        """

        # Pretend we have a MAG - refactor if in future MAGs are implemented
        G = PAG()
        G.add_edge("A", "Ef", G.undirected_edge_name)
        G.add_edge("Ef", "R", G.directed_edge_name)
        G.add_edge("D", "R", G.directed_edge_name)
        G._edge_graphs.pop("circle")

        sample = dummy_sample(G)
        context = self.context_func().variables(data=sample).build()
        oracle = Oracle(G)
        ci_estimator = oracle
        fci = FCI(ci_estimator=ci_estimator, max_iter=np.inf, selection_bias=True)
        fci.learn_graph(sample, context)
        pag = fci.graph_

        expected_pag = PAG()
        expected_pag.add_edge("A", "Ef", expected_pag.circle_edge_name)
        expected_pag.add_edge("Ef", "A", expected_pag.circle_edge_name)
        expected_pag.add_edge("Ef", "R", expected_pag.directed_edge_name)
        expected_pag.add_edge("R", "Ef", expected_pag.circle_edge_name)
        expected_pag.add_edge("R", "D", expected_pag.circle_edge_name)
        expected_pag.add_edge("D", "R", expected_pag.directed_edge_name)
        assert pag.nodes() == expected_pag.nodes()
        assert pag.edges() == expected_pag.edges()
