from typing import Any, Dict, Optional, Set, Union

import networkx as nx
import pandas as pd

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
    _state_variables: Optional[Dict[str, Any]] = None

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
        latents : Optional[Set[Column]] - variables that are latent
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        data : Optional[pd.DataFrame] - the data to use for variable inference.

        Returns
        -------
        ContextBuilder
            The builder instance
        """
        if data is not None:
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

            observed = set(observed)  # type: ignore
            latents = set(latents)  # type: ignore

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
        ContextBuilder
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
        if self._state_variables is None:
            self._state_variables = dict()

        self._state_variables[name] = var
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        Context
            The populated Context object
        """
        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")
        return Context(
            init_graph=self._interpolate_graph(),
            included_edges=self._interpolate_included_edges(),
            excluded_edges=self._interpolate_excluded_edges(),
            variables=self._observed_variables,
            latents=self._latent_variables or set(),
            state_variables=self._state_variables or dict(),
        )

    def _interpolate_graph(self) -> nx.Graph:
        if self._observed_variables is None:
            raise ValueError("Must set variables() before building Context.")

        variables = self._observed_variables or set()
        graph = self._graph
        # initialize the starting graph
        if graph is None:
            return nx.complete_graph(variables, create_using=nx.Graph)
        else:
            if set(graph.nodes) != set(variables):
                raise ValueError(
                    f"The nodes within the initial graph, {graph.nodes}, "
                    f"do not match the nodes in the passed in data, {variables}."
                )
            return graph

    def _interpolate_included_edges(self) -> Union[nx.Graph, nx.DiGraph]:
        # initialize set of fixed and included edges
        return self._included_edges or nx.empty_graph(
            self._observed_variables, create_using=nx.Graph
        )

    def _interpolate_excluded_edges(self) -> Union[nx.Graph, nx.DiGraph]:
        # initialize set of fixed and included edges
        return self._excluded_edges or nx.empty_graph(
            self._observed_variables, create_using=nx.Graph
        )


def make_context(context: Optional[Context] = None) -> ContextBuilder:
    """Create a new ContextBuilder instance.

    Returns
    -------
    ContextBuilder
        The new ContextBuilder instance
    """
    result = ContextBuilder()
    if context is not None:
        result.graph(context.init_graph)
        result.edges(context.included_edges, context.excluded_edges)
        result.variables(context.observed_variables, context.latent_variables)
        result.state_variables(context.state_variables)
    return result
