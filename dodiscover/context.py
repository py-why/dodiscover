from copy import copy, deepcopy
from typing import Optional, Set, Union

import networkx as nx
import pandas as pd

from ._protocol import Graph


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

    Setting the apriori explicit direction of an edge is not supported yet.
    """

    _data: pd.DataFrame
    _variables: Set
    _latents: Set
    _init_graph: Graph
    _included_edges: nx.Graph
    _excluded_edges: nx.Graph

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Optional[Set] = None,
        latents: Optional[Set] = None,
        init_graph: Optional[Graph] = None,
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
            if set(graph.nodes) != set(variables):
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
    def init_graph(self) -> Graph:
        return self._init_graph

    def copy(self):
        """Create a copy of the Context object.

        Returns
        -------
        context : Context
            A copy.
        """
        context = Context(
            data=self._data.copy(deep=True),
            variables=copy(self._variables),
            latents=copy(self._latents),
            init_graph=deepcopy(self._init_graph),
            included_edges=deepcopy(self._included_edges),
            excluded_edges=deepcopy(self._excluded_edges),
        )
        return context
