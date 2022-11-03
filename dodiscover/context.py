import inspect
from typing import Any, Dict, Set, TypeVar

import networkx as nx

from ._protocol import TimeSeriesGraph
from .typing import Column, NetworkxGraph
from .utils import dict_compare


class Context:
    """Context of assumptions, domain knowledge and data.

    This should NOT be instantiated directly. One should instead
    use `dodiscover.make_context` to build a Context data structure.

    Parameters
    ----------
    observed_variables : Set
        Set of observed variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    latent_variables : Set
        Set of latent "unobserved" variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    init_graph : Graph
        The graph to start with.
    included_edges : nx.Graph
        Included edges without direction.
    excluded_edges : nx.Graph
        Excluded edges without direction.
    state_variables : dict
        A dictionary of state variables that are preserved during the
        causal discovery algorithm. For example, the FCI algorithm must store
        the intermediate PAG before running a second phase of skeleton discovery.

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
    _init_graph: nx.Graph
    _included_edges: nx.Graph
    _excluded_edges: nx.Graph
    _state_variables: Dict[str, Any]

    def __init__(
        self,
        observed_variables: Set[Column],
        latent_variables: Set[Column],
        init_graph: nx.Graph,
        included_edges: NetworkxGraph,
        excluded_edges: NetworkxGraph,
        state_variables: Dict[str, Any],
    ) -> None:
        # set to class
        self._state_variables = state_variables
        self._variables = observed_variables
        self._latents = latent_variables
        self._init_graph = init_graph
        self._included_edges = included_edges
        self._excluded_edges = excluded_edges

    @property
    def _internal_graphs(self):
        """Private property to store a list of the names of graph objects."""
        graphs = ["init_graph", "included_edges", "excluded_edges"]
        return graphs

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the context."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the context parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "dodiscover context objects should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this context.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this context and
            contained subobjects that are context.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def __eq__(self, context: "Context") -> bool:
        context_params = context.get_params()
        self_params = self.get_params()

        # graph objects that we must check explicitly
        graph_comps = self._internal_graphs
        context_graphs = []
        self_graphs = []
        for name in graph_comps:
            context_graphs.append(context_params.pop(name))
            self_graphs.append(self_params.pop(name))

        # check all graphs are isomorphic
        for ctx_graph, self_graph in zip(context_graphs, self_graphs):
            if not nx.is_isomorphic(ctx_graph, self_graph):
                return False

        # finally check the rest
        added, removed, modified, _ = dict_compare(context_params, self_params)
        if len(added) > 0 or len(removed) > 0 or len(modified) > 0:
            return False
        return True

    @property
    def included_edges(self) -> nx.Graph:
        return self._included_edges

    @property
    def excluded_edges(self) -> nx.Graph:
        return self._excluded_edges

    @property
    def init_graph(self) -> nx.Graph:
        return self._init_graph

    @property
    def observed_variables(self) -> Set[Column]:
        return self._variables

    @property
    def latent_variables(self) -> Set[Column]:
        return self._latents

    @property
    def state_variables(self) -> Dict[str, Any]:
        return self._state_variables

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

    def state_variable(self, name: str) -> Any:
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


class TimeSeriesContext(Context):
    """Context for time-series causal discovery.

    Parameters
    ----------
    variables : Set
        Set of observed variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set. In a time-series
        context, variables do not have a time-index. See Notes for details.
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
    state_variables : Dict[str, Any]
        _description_
    included_lag_edges : _type_
        _description_
    excluded_lag_edges : _type_
        _description_
    max_lag : int
        _description_
    contemporaneous_edges : bool
        Whether or not to assume contemporaneous edges.
    """

    def __init__(
        self,
        observed_variables: Set[Column],
        latent_variables: Set[Column],
        init_graph: nx.Graph,
        included_edges: NetworkxGraph,
        excluded_edges: NetworkxGraph,
        state_variables: Dict[str, Any],
        included_lag_edges: TimeSeriesGraph,
        excluded_lag_edges: TimeSeriesGraph,
        max_lag: int,
        contemporaneous_edges: bool,
    ) -> None:
        super().__init__(
            observed_variables,
            latent_variables,
            init_graph,
            included_edges,
            excluded_edges,
            state_variables,
        )
        self._max_lag = max_lag
        self._included_lag_edges = included_lag_edges
        self._excluded_lag_edges = excluded_lag_edges
        self._contemporaneous_edges = contemporaneous_edges

    @property
    def _internal_graphs(self):
        """Private property to store a list of the names of graph objects."""
        graphs = [
            "init_graph",
            "included_edges",
            "excluded_edges",
            "included_lag_edges",
            "excluded_lag_edges",
        ]
        return graphs

    @property
    def contemporaneous_edges(self) -> bool:
        return self._contemporaneous_edges

    @property
    def max_lag(self) -> int:
        return self._max_lag

    @property
    def included_lag_edges(self) -> TimeSeriesGraph:
        return self._included_lag_edges

    @property
    def excluded_lag_edges(self) -> TimeSeriesGraph:
        return self._excluded_lag_edges


ContextType = TypeVar("ContextType", bound=Context)
