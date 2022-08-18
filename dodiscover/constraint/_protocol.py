from typing import Dict, FrozenSet, Iterable, Protocol

import networkx as nx


class GraphProtocol(Protocol):
    def nodes(self) -> Iterable:
        pass

    def edges(self) -> Iterable:
        pass

    def has_edge(self, u, v, edge_type) -> bool:
        pass

    def add_node(self, node_for_adding, **attr) -> None:
        pass

    def remove_node(self, u) -> None:
        pass

    def neighbors(self, node) -> Iterable:
        """An iterable for all nodes that have any edge connection with 'node'."""
        pass

    def to_undirected(self) -> nx.Graph:
        pass


class EquivalenceClassProtocol(GraphProtocol, Protocol):
    def orient_uncertain_edge(self, u, v) -> None:
        """Orients an uncertain edge in the equivalence class to directed 'u'*->'v'."""
        pass

    @property
    def excluded_triples(self) -> Dict[FrozenSet, None]:
        """A set of triples that are excluded from orientation."""
        pass
