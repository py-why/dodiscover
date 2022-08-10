from typing import Dict, FrozenSet, Iterable, Protocol

import networkx as nx


class GraphProtocol(Protocol):
    def nodes(self) -> Iterable:
        pass

    def edges(self) -> Iterable:
        pass

    def add_node(self, node_for_adding, **attr):
        pass

    def remove_node(self, u):
        pass

    def neighbors(self, node) -> Iterable:
        """An iterable for all nodes that have any edge connection with 'node'."""
        pass

    def to_undirected(self) -> nx.Graph:
        pass


class EquivalenceClassProtocol(Protocol, GraphProtocol):
    def orient_uncertain_edge(self, u, v):
        """Orients an uncertain edge in the equivalence class."""
        pass

    @property
    def excluded_triples(self) -> Dict[FrozenSet]:
        """A set of triples that are excluded from orientation."""
        pass
