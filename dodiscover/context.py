from copy import copy, deepcopy
from typing import Any, Dict, Set, Union

import networkx as nx

from ._protocol import Graph
from .typing import Column


class Context:
    """Context of assumptions, domain knowledge and data.

    Parameters
    ----------
    variables : Set
        Set of observed variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    latents : Set
        Set of latent "unobserved" variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    init_graph : Graph
        The graph to start with.
    included_edges : nx.Graph
        Included edges without direction.
    excluded_edges : nx.Graph
        Excluded edges without direction.

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
    _state_variables: Dict[str, Any]

    def __init__(
        self,
        variables: Set[Column],
        latents: Set[Column],
        init_graph: Graph,
        included_edges: Union[nx.Graph, nx.DiGraph],
        excluded_edges: Union[nx.Graph, nx.DiGraph],
        state_variables: Dict[str, Any],
    ) -> None:
        # set to class
        self._state_variables = state_variables
        self._variables = variables
        self._latents = latents
        self._init_graph = init_graph
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

    def add_state_variable(self, name: str, var: Any) -> None:
        """Add a state variable.

        Called by an algorithm to persist data objects that
        are used in intermediate steps.

        Parameters
        ----------
        name : str
            The name of the state variable.
        var : any
            Any state variable.
        """
        self._state_variables[name] = var

    def get_state_variable(self, name: str) -> Any:
        """Get a state variable.

        Parameters
        ----------
        name : str
            The name of the state variable.

        Returns
        -------
        state_var : Any
            The state variable.
        """
        if name not in self._state_variables:
            raise RuntimeError(f"{name} is not a state variable: {self._state_variables}")

        return self._state_variables[name]

    def copy(self):
        """Create a copy of the Context object.

        Performs a deep-copy when necessary.

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
            state_variables=deepcopy(self._state_variables),
        )
        return context
