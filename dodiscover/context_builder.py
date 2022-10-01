from copy import copy, deepcopy
from typing import Any, Dict, Optional, Set, Tuple, cast

import networkx as nx
import pandas as pd

from ._protocol import Graph
from .context import Context
from .typing import Column, NetworkxGraph


class ContextBuilder:
    """A builder class for creating Context objects ergonomically.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data. This should NOT be instantiated directly. One should instead use
    `dodiscover.make_context` to build a Context data structure.
    """

    _graph: Optional[Graph] = None
    _included_edges: Optional[NetworkxGraph] = None
    _excluded_edges: Optional[NetworkxGraph] = None
    _observed_variables: Optional[Set[Column]] = None
    _latent_variables: Optional[Set[Column]] = None
    _state_variables: Dict[str, Any] = dict()

    def graph(self, graph: Graph) -> "ContextBuilder":
        """Set the partial graph to start with.

        Parameters
        ----------
        graph : Graph
            The new graph instance.

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._graph = graph
        return self

    def edges(
        self,
        include: Optional[NetworkxGraph] = None,
        exclude: Optional[NetworkxGraph] = None,
    ) -> "ContextBuilder":
        """Set edge constraints to apply in discovery.

        Parameters
        ----------
        included : Optional[NetworkxGraph]
            Edges that should be included in the resultant graph
        excluded : Optional[NetworkxGraph]
            Edges that must be excluded in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._included_edges = include
        self._excluded_edges = exclude
        return self

    def variables(
        self,
        observed: Optional[Set[Column]] = None,
        latents: Optional[Set[Column]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "ContextBuilder":
        """Set variable-list information to utilize in discovery.

        Parameters
        ----------
        observed : Optional[Set[Column]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        latents : Optional[Set[Column]]
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        data : Optional[pd.DataFrame]
            the data to use for variable inference.

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._observed_variables = observed
        self._latent_variables = latents

        if data is not None:
            (observed, latents) = self._interpolate_variables(data, observed, latents)
            self._observed_variables = observed
            self._latent_variables = latents

        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")

        return self

    def state_variables(self, state_variables: Dict[str, Any]) -> "ContextBuilder":
        """Set the state variables to use in discovery.

        Parameters
        ----------
        state_variables : Dict[str, Any]
            The state variables to use in discovery.

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._state_variables = state_variables
        return self

    def state_variable(self, name: str, var: Any) -> "ContextBuilder":
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
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        context : Context
            The populated Context object.
        """
        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")

        empty_graph = lambda: nx.empty_graph(self._observed_variables, create_using=nx.Graph)
        return Context(
            init_graph=self._interpolate_graph(),
            included_edges=self._included_edges or empty_graph(),
            excluded_edges=self._excluded_edges or empty_graph(),
            variables=self._observed_variables,
            latents=self._latent_variables or set(),
            state_variables=self._state_variables,
        )

    def _interpolate_variables(
        self,
        data: pd.DataFrame,
        observed: Optional[Set[Column]] = None,
        latents: Optional[Set[Column]] = None,
    ) -> Tuple[Set[Column], Set[Column]]:
        # initialize and parse the set of variables, latents and others
        columns = set(data.columns)
        if observed is not None and latents is not None:
            if columns - set(observed) != set(latents):
                raise ValueError(
                    "If observed and latents are set, then they must be "
                    "include all columns in data."
                )
        elif observed is None and latents is not None:
            observed = columns - set(latents)
        elif latents is None and observed is not None:
            latents = columns - set(observed)
        elif observed is None and latents is None:
            # when neither observed, nor latents is set, it is assumed
            # that the data is all "not latent"
            observed = columns
            latents = set()

        observed = set(cast(Set[Column], observed))
        latents = set(cast(Set[Column], latents))
        return (observed, latents)

    def _interpolate_graph(self) -> nx.Graph:
        if self._observed_variables is None:
            raise ValueError("Must set variables() before building Context.")

        complete_graph = lambda: nx.complete_graph(self._observed_variables, create_using=nx.Graph)
        has_all_variables = lambda g: set(g.nodes) == set(self._observed_variables)

        # initialize the starting graph
        if self._graph is None:
            return complete_graph()
        else:
            if not has_all_variables(self._graph):
                raise ValueError(
                    f"The nodes within the initial graph, {self._graph.nodes}, "
                    f"do not match the nodes in the passed in data, {self._observed_variables}."
                )
            return self._graph


def make_context(context: Optional[Context] = None) -> ContextBuilder:
    """Create a new ContextBuilder instance.

    Returns
    -------
    result : ContextBuilder
        The new ContextBuilder instance

    Examples
    --------
    This creates a context object denoting that there are three observed
    variables, ``(1, 2, 3)``.
    >>> context_builder = make_context()
    >>> context = context_builder.variables([1, 2, 3]).build()
    """
    result = ContextBuilder()
    if context is not None:
        result.graph(deepcopy(context.init_graph))
        result.edges(deepcopy(context.included_edges), deepcopy(context.excluded_edges))
        result.variables(copy(context.observed_variables), copy(context.latent_variables))
        result.state_variables(deepcopy(context.state_variables))
    return result
