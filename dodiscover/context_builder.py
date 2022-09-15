from typing import Optional, Set, Union

import networkx as nx
import pandas as pd

from ._protocol import Graph
from .context import Context


class ContextBuilder:
    """A builder class for creating Context objects ergonomically.

    The context builder provides a way to capture assumptions, domain knowledge, and data.

    Parameters
    ----------
    data : pd.DataFrame
        A dataset in data-frame form, consisting of samples as rows and variables as columns.
    """

    _graph: Optional[Graph] = None
    _included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None
    _excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None
    _observed_variables: Optional[Set[str]] = None
    _latent_variables: Optional[Set[str]] = None

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def data(self, data: pd.DataFrame) -> "ContextBuilder":
        """Set the dataset to use.

        Parameters
        ----------
        data : pd.DataFrame - the new dataframe to use

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._data = data
        return self

    def graph(self, graph: Graph) -> "ContextBuilder":
        """Set the partial graph to start with.

        Parameters
        ----------
        data : pd.DataFrame - the new dataframe to use


        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._graph = graph
        return self

    def edge_constraints(
        self,
        included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
        excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    ) -> "ContextBuilder":
        """Set edge constraints to apply in discovery.

        Parameters
        ----------
        included_edges : Optional[Union[nx.Graph, nx.DiGraph]]
            Edges that should be included in the resultant graph
        excluded_edges : Optional[Union[nx.Graph, nx.DiGraph]]
            Edges that must be excluded in the resultant graph

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._included_edges = included_edges
        self._excluded_edges = excluded_edges
        return self

    def features(
        self, observed_variables: Optional[Set[str]], latent_variables: Optional[Set[str]]
    ) -> "ContextBuilder":
        """Set feature-list information to utilize in discovery.

        Parameters
        ----------
        observed_variables : Optional[Set[str]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        latent_variables : Optional[Set[str]] - variables that are latent
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        self._observed_variables = observed_variables
        self._latent_variables = latent_variables
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        Context
            The populated Context object
        """
        return Context(
            data=self._data,
            init_graph=self._graph,
            included_edges=self._included_edges,
            excluded_edges=self._excluded_edges,
            variables=self._observed_variables,
            latents=self._latent_variables,
        )


def make_context(data: pd.DataFrame) -> ContextBuilder:
    """Create a new ContextBuilder instance.

    Parameters
    ----------
    data : pd.DataFrame
        The data to use in the context

    Returns
    -------
    ContextBuilder
        The new ContextBuilder instance
    """
    return ContextBuilder(data)
