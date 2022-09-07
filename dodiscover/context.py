from typing import Optional, Set, Union

import networkx as nx
import pandas as pd

from ._protocol import GraphProtocol


class ContextBuilder:
    """A builder class that allows users to create Context objects ergonomically.
    Context objects contain assumptions, domain knowledge, and data.

    Parameters
    ----------
    data : pd.DataFrame
        A dataset in data-frame form, consisting of samples as rows and variables as columns.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data
        self._graph = None
        self._included_edges = None
        self._excluded_edges = None
        self._observed_variables = None
        self._latent_variables = None

    """
    Initialize the data

    Parameters
    ----------
    data : pd.DataFrame - the new dataframe to use
    """

    def data(self, data: pd.DataFrame) -> "ContextBuilder":
        self._data = data
        return self

    """
    Initialize the data

    Parameters
    ----------
    data : pd.DataFrame - the new dataframe to use

    
    Returns
    -------
    The builder instance
    """

    def init_graph(self, graph: GraphProtocol) -> "ContextBuilder":
        self._graph = graph
        return self

    """
    Set edge constraints to apply in discovery

    Parameters
    ----------
    included_edges : Optional[Union[nx.Graph, nx.DiGraph]]
        Edges that should be included in the resultant graph
    excluded_edges : Optional[Union[nx.Graph, nx.DiGraph]]
        Edges that must be excluded in the resultant graph

    Returns
    -------
    The builder instance
    """

    def edge_constraints(
        self,
        included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
        excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    ) -> "ContextBuilder":
        self._included_edges = included_edges
        self._excluded_edges = excluded_edges
        return self

    """
    Set feature-list information to utilize in discovery

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
    """

    def features(
        self, observed_variables: Set[str], latent_variables: Set[str]
    ) -> "ContextBuilder":
        self._features = features
        self.observed_variables = observed_variables
        self.latent_variables = latent_variables
        return self

    """
    Builds the resultant Context object to be used in discovery tasks

    Returns
    -------
    The populated Context object
    """

    def build() -> Context:
        return Context(
            data=self._data,
            init_graph=self._graph,
            included_edges=self._included_edges,
            excluded_edges=self._excluded_edges,
            variables=self._observed_variables,
            latents=self._latent_variables,
        )


"""Factory function for creating a new ContextBuilder instance

Parameters
----------
data : pd.DataFrame
    The data to use in the context

Returns
-------
The new ContextBuilder instance
"""


def context_builder(data: pd.DataFrame) -> ContextBuilder:
    return ContextBuilder(data)


class Context:
    """Context of assumptions, domain knowledge and data.

    Parameters
    ----------
    data : pd.DataFrame
        A dataset, consisting of samples as rows and columns as variables.
    variables : Optional[Set], optional
        Set of observed variables, by default None. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    latents : Optional[Set], optional
        Set of latent "unobserved" variables, by default None. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    init_graph : Optional[GraphProtocol], optional
        The graph to start with, by default None.
    included_edges : Optional[nx.Graph], optional
        Included edges without direction, by default None.
    excluded_edges : Optional[nx.Graph], optional
        Excluded edges without direction, by default None.

    Raises
    ------
    ValueError
        ``variables`` and ``latents`` if both set, should contain the set of
        all columns in ``data``.

    Notes
    -----
    Context is a data structure for storing assumptions, domain knowledge,
    priors and other structured contexts alongside the datasets. This class
    is used in conjunction with a discovery algorithm.

    Setting the a priori explicit direction of an edge is not supported yet.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Optional[Set] = None,
        latents: Optional[Set] = None,
        init_graph: Optional[GraphProtocol] = None,
        included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
        excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    ) -> None:
        # initialize and parse the set of variables, latents and others
        columns = set(data.columns)
        if variables is not None and latents is not None:
            if columns - set(variables) != set(latents):
                raise ValueError(
                    "If variables and latents are set, then they must be "
                    "include all columns in data."
                )
        elif variables is None and latents is not None:
            variables = columns - set(latents)
        elif latents is None and variables is not None:
            latents = columns - set(variables)
        elif variables is None and latents is None:
            # when neither variables, nor latents is set, it is assumed
            # that the data is all "not latent"
            variables = columns
            latents = set()
        variables = set(variables)  # type: ignore
        latents = set(latents)  # type: ignore

        # initialize the starting graph
        if init_graph is None:
            graph = nx.complete_graph(variables, create_using=nx.Graph)
        else:
            graph = init_graph
            if graph.nodes != variables:
                raise ValueError(
                    f"The nodes within the initial graph, {graph.nodes}, "
                    f"do not match the nodes in the passed in data, {variables}."
                )

        # initialize set of fixed and included edges
        if included_edges is None:
            included_edges = nx.empty_graph(variables, create_using=nx.Graph)
        if excluded_edges is None:
            excluded_edges = nx.empty_graph(variables, create_using=nx.Graph)

        # set to class
        self._data = data
        self._variables = variables
        self._latents = latents
        self._init_graph = graph
        self._included_edges = included_edges
        self._excluded_edges = excluded_edges

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def included_edges(self) -> nx.Graph:
        return self._included_edges

    @property
    def excluded_edges(self) -> nx.Graph:
        return self._excluded_edges

    @property
    def init_graph(self) -> GraphProtocol:
        return self._init_graph
