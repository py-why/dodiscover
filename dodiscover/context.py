from copy import copy, deepcopy
from typing import Optional, Set, Union

import networkx as nx

from ._protocol import Graph
from .typing import Column


class Context:
    """Context of assumptions, domain knowledge and data.

    Parameters
    ----------
    variables : Optional[Set[Column]], optional
        Set of observed variables, by default None. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    latents : Optional[Set[Column]], optional
        Set of latent "unobserved" variables, by default None. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    init_graph : Optional[Graph], optional
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

    _variables: Set[Column]
    _latents: Set[Column]
    _init_graph: Graph
    _included_edges: nx.Graph
    _excluded_edges: nx.Graph

    def __init__(
        self,
        variables: Set[Column],
        latents: Set[Column],
        init_graph: Optional[Graph] = None,
        included_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
        excluded_edges: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    ) -> None:
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
        self._variables = variables
        self._latents = latents
        self._init_graph = graph
        self._included_edges = included_edges
        self._excluded_edges = excluded_edges

    @property
    def included_edges(self) -> nx.Graph:
        return self._included_edges

    @property
    def excluded_edges(self) -> nx.Graph:
        return self._excluded_edges

    @property
    def init_graph(self) -> Graph:
        return self._init_graph

    @property
    def observed_variables(self) -> Set[Column]:
        return self._variables

    @property
    def latent_variables(self) -> Set[Column]:
        return self._latents

    def copy(self):
        """Create a copy of the Context object.

        Returns
        -------
        context : Context
            A copy.
        """
        context = Context(
            variables=copy(self._variables),
            latents=copy(self._latents),
            init_graph=deepcopy(self._init_graph),
            included_edges=deepcopy(self._included_edges),
            excluded_edges=deepcopy(self._excluded_edges),
        )
        return context
