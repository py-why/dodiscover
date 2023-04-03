from copy import deepcopy
from typing import Any, Dict, Optional, Set, Tuple, cast

import networkx as nx
import pandas as pd

from ._protocol import TimeSeriesGraph
from .context import Context, TimeSeriesContext
from .typing import Column, NetworkxGraph


class ContextBuilder:
    """A builder class for creating Context objects ergonomically.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data. This should NOT be instantiated directly. One should instead use
    `dodiscover.make_context` to build a Context data structure.
    """

    _graph: Optional[nx.Graph] = None
    _included_edges: Optional[NetworkxGraph] = None
    _excluded_edges: Optional[NetworkxGraph] = None
    _observed_variables: Optional[Set[Column]] = None
    _latent_variables: Optional[Set[Column]] = None
    _state_variables: Dict[str, Any] = dict()

    def init_graph(self, graph: nx.Graph) -> "ContextBuilder":
        """Set the initial partial undirected graph to start with.

        For example, this could be the complete graph to start with, if there is
        no prior knowledge. Or this could be a graph that is a continuation of a
        previous causal discovery algorithm.

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

    def excluded_edges(self, edges: NetworkxGraph) -> "ContextBuilder":
        """Set excluded non-directional edge constraints to apply in discovery.

        Parameters
        ----------
        edges : Optional[NetworkxGraph]
            Edges that must be excluded in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._excluded_edges = edges
        return self

    def included_edges(self, edges: NetworkxGraph) -> "ContextBuilder":
        """Set included non-directional edge constraints to apply in discovery.

        Parameters
        ----------
        edges : Optional[NetworkxGraph]
            Edges that must be included in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._included_edges = edges
        return self

    def latent_variables(self, latents: Set[Column]):
        """Latent variables."""
        self._latent_variables = latents
        return self

    def observed_variables(self, observed: Set[Column]):
        """Observed variables."""
        self._observed_variables = observed
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

        # initialize an empty graph object as the default for included/excluded edges
        empty_graph = lambda: nx.empty_graph(self._observed_variables, create_using=nx.Graph)

        return Context(
            init_graph=self._interpolate_graph(),
            included_edges=self._included_edges or empty_graph(),
            excluded_edges=self._excluded_edges or empty_graph(),
            observed_variables=self._observed_variables,
            latent_variables=self._latent_variables or set(),
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


class TimeSeriesContextBuilder(ContextBuilder):
    """A builder class for creating TimeSeriesContext objects ergonomically.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data relevant for time-series. This should NOT be instantiated directly.
    One should instead use `dodiscover.make_ts_context` to build a TimeSeriesContext
    data structure.
    """

    _contemporaneous_edges: Optional[bool] = None
    _max_lag: Optional[int] = None
    _included_lag_edges: Optional[TimeSeriesGraph] = None
    _exccluded_lag_edges: Optional[TimeSeriesGraph] = None

    def _interpolate_graph(self) -> nx.Graph:
        from pywhy_graphs.classes import StationaryTimeSeriesGraph
        from pywhy_graphs.classes.timeseries import complete_ts_graph

        if self._observed_variables is None:
            raise ValueError("Must set variables() before building Context.")
        if self._max_lag is None:
            raise ValueError("Must set max_lag before building Context.")

        # initialize the starting graph
        if self._graph is None:
            include_contemporaneous = self._contemporaneous_edges or True
            # create a complete graph over all nodes and time points
            complete_graph = complete_ts_graph(
                variables=self._observed_variables,
                max_lag=self._max_lag,
                include_contemporaneous=include_contemporaneous,
                create_using=StationaryTimeSeriesGraph,
            )
            return complete_graph
        else:
            if set(self._graph.variables) != set(self._observed_variables):
                raise ValueError(
                    f"The nodes within the initial graph, {self._graph.nodes}, "
                    f"do not match the nodes in the passed in data, {self._observed_variables}."
                )

            for var_name in self._observed_variables:
                for lag in range(self._max_lag + 1):
                    if (var_name, -lag) not in self._graph.nodes:
                        raise RuntimeError(
                            f"Graph does not contain all possible nodes, "
                            f"such as {(var_name, -lag)}."
                        )

            return self._graph

    def init_graph(self, graph: TimeSeriesGraph) -> "ContextBuilder":
        """Set the initial time-series graph to begin causal discovery with."""
        return super().init_graph(graph)

    def max_lag(self, lag: int) -> "ContextBuilder":
        """Set the maximum time lag."""
        if lag <= 0:
            raise ValueError(f"Lag in time-series graphs should be > 0, not {lag}.")
        self._max_lag = lag
        return self

    def contemporaneous_edges(self, present: bool) -> "ContextBuilder":
        """Whether or not to assume contemporaneous edges."""
        self._contemporaneous_edges = present
        return self

    def included_lag_edges(self, edges: TimeSeriesGraph) -> "ContextBuilder":
        """Apriori set lagged edges."""
        self._included_lag_edges = edges
        return self

    def excluded_lag_edges(self, edges: TimeSeriesGraph) -> "ContextBuilder":
        """Apriori excluded lagged edges."""
        self._excluded_lag_edges = edges
        return self

    def build(self) -> TimeSeriesContext:
        """Build a time-series context object."""
        from pywhy_graphs.classes.timeseries import empty_ts_graph

        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")

        if self._max_lag is None:
            raise ValueError("Max lag must be set before building time-series context")

        # initialize an empty graph object as the default for included/excluded edges
        empty_graph = lambda: nx.empty_graph(self._observed_variables, create_using=nx.Graph)
        empty_time_graph = lambda: empty_ts_graph(self._observed_variables, max_lag=self._max_lag)

        # initialize assumption of contemporaneous edges by default
        return TimeSeriesContext(
            init_graph=self._interpolate_graph(),
            included_edges=self._included_edges or empty_graph(),
            excluded_edges=self._excluded_edges or empty_graph(),
            observed_variables=self._observed_variables,
            latent_variables=self._latent_variables or set(),
            state_variables=self._state_variables,
            included_lag_edges=self._included_lag_edges or empty_time_graph(),
            excluded_lag_edges=self._included_lag_edges or empty_time_graph(),
            max_lag=self._max_lag,
            contemporaneous_edges=self._contemporaneous_edges or True,
        )


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
        params = context.get_params()
        for param, value in params.items():
            if not hasattr(result, param):
                raise RuntimeError(f"{param} is not a member of Context and ContexBuilder.")
            # get the function for parameter
            getattr(result, param)(deepcopy(value))
    return result


def make_ts_context(context: Optional[Context] = None) -> TimeSeriesContextBuilder:
    """Create a time-series context builder."""
    result = TimeSeriesContextBuilder()
    if context is not None:
        params = context.get_params()
        for param, value in params.items():
            if not hasattr(result, param):
                raise RuntimeError(f"{param} is not a member of Context and ContexBuilder.")
            # get the function for parameter
            getattr(result, param)(deepcopy(value))
    return result
