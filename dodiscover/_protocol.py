from typing import Dict, FrozenSet, Iterable, Mapping, Protocol

import networkx as nx


class GraphProtocol(Protocol):
    """Protocol for graphs to work with dodiscover algorithms."""

    @property
    def nodes(self) -> Mapping:
        """Return an iterable over nodes in graph."""
        pass

    @property
    def edges(self) -> Mapping:
        """Return an iterable over edge tuples in graph."""
        pass

    def has_edge(self, u, v, edge_type: str) -> bool:
        """Check if graph has an edge for a specific edge type."""
        pass

    def add_node(self, node, **attr) -> None:
        """Add a node to the graph."""
        pass

    def remove_node(self, u) -> None:
        """Remove a node from the graph."""
        pass

    def remove_edges_from(self, edges) -> None:
        """Remove an iterable of edges from graph."""
        pass

    def neighbors(self, node) -> Iterable:
        """Iterate over all nodes that have any edge connection with 'node'."""
        pass

    def to_undirected(self) -> nx.Graph:
        """Convert a graph to a fully undirected networkx graph.

        All nodes are connected by an undirected edge if there are any
        edges between the two.
        """
        pass


class EquivalenceClassProtocol(GraphProtocol, Protocol):
    """Protocol for equivalence class of graphs."""

    @property
    def excluded_triples(self) -> Dict[FrozenSet, None]:
        """A set of triples that are excluded from orientation."""
        pass

    @property
    def directed_edge_name(self) -> str:
        """Name of the directed edges."""
        pass

    @property
    def undirected_edge_name(self) -> str:
        """Name of the undirected edges."""
        pass

    def orient_uncertain_edge(self, u, v) -> None:
        """Orients an uncertain edge in the equivalence class to directed ``'u'*->'v'``."""
        pass

    def mark_unfaithful_triple(self, v_i, u, v_j) -> None:
        """Mark a triple as unfaithful, and put it in the excluded triple set."""
        pass

    def predecessors(self, node) -> Iterable:
        """Nodes with directed edges pointing to 'node'."""
        pass

    def successors(self, node) -> Iterable:
        """Nodes with directed edges pointing from 'node'."""
        pass