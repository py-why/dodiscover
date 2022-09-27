from typing import Optional, Set, Union

import networkx as nx

from ._protocol import Graph
from .context import Context
from .typing import Column


class ContextBuilder:
    """A builder class for creating Context objects ergonomically.

    The context builder provides a way to capture assumptions, domain knowledge, and data.
    """

    _graph: Optional[Graph] = None
    _included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None
    _excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None
    _observed_variables: Optional[Set[Column]] = None
    _latent_variables: Optional[Set[Column]] = None

    def graph(self, graph: Graph) -> "ContextBuilder":
        """Set the partial graph to start with.

        Parameters
        ----------
        graph : Graph - the new graph instance


        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._graph = graph
        return self

    def edges(
        self,
        include: Optional[Union[nx.Graph, nx.DiGraph]] = None,
        exclude: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    ) -> "ContextBuilder":
        """Set edge constraints to apply in discovery.

        Parameters
        ----------
        included : Optional[Union[nx.Graph, nx.DiGraph]]
            Edges that should be included in the resultant graph
        excluded : Optional[Union[nx.Graph, nx.DiGraph]]
            Edges that must be excluded in the resultant graph

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._included_edges = include
        self._excluded_edges = exclude
        return self

    def variables(
        self, observed: Optional[Set[Column]], latent: Optional[Set[Column]]
    ) -> "ContextBuilder":
        """Set variable-list information to utilize in discovery.

        Parameters
        ----------
        observed : Optional[Set[Column]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        latent : Optional[Set[Column]] - variables that are latent
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._observed_variables = observed
        self._latent_variables = latent
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        Context
            The populated Context object
        """
        return Context(
            init_graph=self._graph,
            included_edges=self._included_edges,
            excluded_edges=self._excluded_edges,
            variables=self._observed_variables,
            latents=self._latent_variables,
        )


def make_context() -> ContextBuilder:
    """Create a new ContextBuilder instance.

    Returns
    -------
    ContextBuilder
        The new ContextBuilder instance
    """
    return ContextBuilder()
