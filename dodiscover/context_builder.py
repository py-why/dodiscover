import types
from copy import copy
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd

from ._protocol import Graph
from .context import Context
from .typing import Column, NetworkxGraph

CALLABLES = types.FunctionType, types.MethodType


class ContextBuilder:
    """A builder class for creating observational data Context objects ergonomically.

    The ContextBuilder is meant solely to build Context objects that work
    with observational datasets.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data. This should NOT be instantiated directly. One should instead use
    `dodiscover.make_context` to build a Context data structure.
    """

    _init_graph: Optional[Graph] = None
    _included_edges: Optional[NetworkxGraph] = None
    _excluded_edges: Optional[NetworkxGraph] = None
    _observed_variables: Optional[Set[Column]] = None
    _latent_variables: Optional[Set[Column]] = None
    _state_variables: Dict[str, Any] = dict()

    def __init__(self) -> None:
        # perform an error-check on subclass definitions of ContextBuilder
        for attribute, value in self.__class__.__dict__.items():
            if isinstance(value, CALLABLES) or isinstance(value, property):
                continue
            if attribute.startswith("__"):
                continue

            if not hasattr(self, attribute[1:]):
                raise RuntimeError(
                    f"Context objects has class attributes that do not have "
                    f"a matching class method to set the attribute, {attribute}. "
                    f"The form of the attribute must be '_<name>' and a "
                    f"corresponding function name '<name>'."
                )

    def init_graph(self, graph: Graph) -> "ContextBuilder":
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
        self._init_graph = graph
        return self

    def excluded_edges(self, exclude: Optional[NetworkxGraph]) -> "ContextBuilder":
        """Set exclusion edge constraints to apply in discovery.

        Parameters
        ----------
        excluded : Optional[NetworkxGraph]
            Edges that should be excluded in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        if self._included_edges is not None:
            for u, v in exclude.edges:  # type: ignore
                if self._included_edges.has_edge(u, v):
                    raise RuntimeError(f"{(u, v)} is already specified as an included edge.")
        self._excluded_edges = exclude
        return self

    def included_edges(self, include: Optional[NetworkxGraph]) -> "ContextBuilder":
        """Set inclusion edge constraints to apply in discovery.

        Parameters
        ----------
        included : Optional[NetworkxGraph]
            Edges that should be included in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        if self._excluded_edges is not None:
            for u, v in include.edges:  # type: ignore
                if self._excluded_edges.has_edge(u, v):
                    raise RuntimeError(f"{(u, v)} is already specified as an excluded edge.")
        self._included_edges = include
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

    def observed_variables(self, observed: Optional[Set[Column]] = None) -> "ContextBuilder":
        """Set observed variables.

        Parameters
        ----------
        observed : Optional[Set[Column]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        """
        if self._latent_variables is not None and any(
            obs_var in self._latent_variables for obs_var in observed  # type: ignore
        ):
            raise RuntimeError(
                f"Latent variables are set already {self._latent_variables}, "
                f'which contain variables you are trying to set as "observed".'
            )
        self._observed_variables = observed
        return self

    def latent_variables(self, latents: Optional[Set[Column]] = None) -> "ContextBuilder":
        """Set latent variables.

        Parameters
        ----------
        latents : Optional[Set[Column]]
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        """
        if self._observed_variables is not None and any(
            latent_var in self._observed_variables for latent_var in latents  # type: ignore
        ):
            raise RuntimeError(
                f"Observed variables are set already {self._observed_variables}, "
                f'which contain variables you are trying to set as "latent".'
            )
        self._latent_variables = latents
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

        empty_graph = self._empty_graph_func(self._observed_variables)
        return Context(
            init_graph=self._interpolate_graph(self._observed_variables),
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
                    "If observed and latents are both set, then they must "
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

    def _interpolate_graph(self, graph_variables) -> nx.Graph:
        if self._observed_variables is None:
            raise ValueError("Must set variables() before building Context.")

        complete_graph = lambda: nx.complete_graph(graph_variables, create_using=nx.Graph)
        has_all_variables = lambda g: set(g.nodes).issuperset(set(self._observed_variables))

        # initialize the starting graph
        if self._init_graph is None:
            return complete_graph()
        else:
            if not has_all_variables(self._init_graph):
                raise ValueError(
                    f"The nodes within the initial graph, {self._init_graph.nodes}, "
                    f"do not match the nodes in the passed in data, {self._observed_variables}."
                )
            return self._init_graph

    def _empty_graph_func(self, graph_variables) -> Callable:
        empty_graph = lambda: nx.empty_graph(graph_variables, create_using=nx.Graph)
        return empty_graph


class InterventionalContextBuilder(ContextBuilder):
    """A builder class for creating observational+interventional data Context objects.

    The InterventionalContextBuilder is meant solely to build Context objects that work
    with observational + interventional datasets.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data. This should NOT be instantiated directly. One should instead use
    :func:`dodiscover.make_context` to build a Context data structure.

    Notes
    -----
    The number of distributions and/or interventional targets must be set in order
    to build the :class:`~.context.Context` object here.
    """

    _intervention_targets: Optional[List[Tuple[Column]]] = None
    _num_distributions: Optional[int] = None
    _obs_distribution: bool = True

    def obs_distribution(self, has_obs_distrib: bool):
        """Whether or not we have access to the observational distribution.

        By default, this is True and assumed to be the first distribution.
        """
        self._obs_distribution = has_obs_distrib
        return self

    def num_distributions(self, num_distribs: int):
        """Set the number of data distributions we are expected to have access to.

        Note this must include observational too if observational is assumed present.
        To assume that we do not have access to observational data, use the
        :meth:`InterventionalContextBuilder.obs_distribution` to turn off that assumption.

        Parameters
        ----------
        num_distribs : int
            Number of distributions we will have access to. Will set the number of
            distributions to be ``num_distribs + 1`` if ``_obs_distribution is True`` (default).
        """
        self._num_distributions = num_distribs
        return self

    def intervention_targets(self, targets: List[Tuple[Column]]):
        """Set known intervention targets of the data.

        Will also automatically infer the F-nodes that will be present
        in the graph. For more information on F-nodes see ``pywhy-graphs``.

        Parameters
        ----------
        interventions : List of tuple
            A list of tuples of nodes that are known intervention targets.
            Assumes that the order of the interventions marked are those of the
            passed in the data.

            If intervention targets are unknown, then this is not necessary.
        """
        self._intervention_targets = targets
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
        if self._num_distributions is None:
            warn(
                "There is no intervention context set. Are you sure you are using "
                "the right contextbuilder? If you only have observational data "
                "use `ContextBuilder` instead of `InterventionContextBuilder`."
            )

        # infer intervention targets and number of distributions
        if self._intervention_targets is None:
            intervention_targets = []
        else:
            intervention_targets = self._intervention_targets
        if self._num_distributions is None:
            num_distributions = int(self._obs_distribution) + len(intervention_targets)
        else:
            num_distributions = self._num_distributions

        # error-check if intervention targets was set that it matches the distributions
        if len(intervention_targets) > 0:
            if len(intervention_targets) + int(self._obs_distribution) != num_distributions:
                raise RuntimeError(
                    f"Setting the number of distributions {num_distributions} does not match the "
                    f"number of intervention targets {len(intervention_targets)}."
                )

        # get F-nodes and sigma-map
        f_nodes, sigma_map, symmetric_diff_map = self._create_augmented_nodes(
            intervention_targets, num_distributions
        )
        graph_variables = set(self._observed_variables).union(set(f_nodes))

        empty_graph = self._empty_graph_func(graph_variables)
        return Context(
            init_graph=self._interpolate_graph(graph_variables),
            included_edges=self._included_edges or empty_graph(),
            excluded_edges=self._excluded_edges or empty_graph(),
            observed_variables=self._observed_variables,
            latent_variables=self._latent_variables or set(),
            state_variables=self._state_variables,
            intervention_targets=intervention_targets,
            f_nodes=f_nodes,
            sigma_map=sigma_map,
            symmetric_diff_map=symmetric_diff_map,
            obs_distribution=self._obs_distribution,
            num_distributions=num_distributions,
        )

    def _create_augmented_nodes(
        self, intervention_targets, num_distributions
    ) -> Tuple[List, Dict, Dict]:
        """Create augmented nodes, sigma map and optionally a symmetric difference map.

        Given a number of distributions attributed to interventions, one constructs
        F-nodes to add to the causal graph via one of two procedures:

        - (known targets): For all pairs of intervention targets, form the
          symmetric difference and then assign this to a new F-node.
          This is ``n_targets choose 2``
        - (unknown targets): For all pairs of incoming distributions, form
          a new F-node. This is ``n_distributions choose 2``

        The difference is the additional information is encoded in the known
        targets case. That is we know the symmetric difference mapping for each
        F-node.

        Returns
        -------
        Tuple[List, Dict[Any, Tuple], Dict[Any, FrozenSet]]
            _description_
        """
        augmented_nodes = []
        sigma_map = dict()
        symmetric_diff_map = dict()

        # add the empty intervention if there is assumed observational data
        if self._obs_distribution:
            distribution_targets_idx = [0]
        else:
            distribution_targets_idx = []

        # now map all distribution targets to their indexed distribution
        int_dist_idx = np.arange(int(self._obs_distribution), num_distributions).tolist()
        distribution_targets_idx.extend(int_dist_idx)

        # store known-targets, which are sets of nodes
        targets = []
        if len(intervention_targets) > 0:
            if self._obs_distribution:
                targets.append(())
            targets.extend(copy(list(intervention_targets)))  # type: ignore

        # create F-nodes, their symmetric difference mapping and sigma-mapping to
        # intervention targets
        for idx, (jdx, kdx) in enumerate(combinations(distribution_targets_idx, 2)):
            f_node = ("F", idx)
            augmented_nodes.append(f_node)
            sigma_map[f_node] = (jdx, kdx)

            # if we additionally know the intervention targets
            if len(intervention_targets) > 0:
                i_target: Set = set(targets[jdx])
                j_target: Set = set(targets[kdx])

                # form symmetric difference and store its frozenset
                # (so that way order is not important)
                f_node_targets = frozenset(i_target.symmetric_difference(j_target))
                symmetric_diff_map[f_node] = f_node_targets
        return augmented_nodes, sigma_map, symmetric_diff_map

    def _interpolate_graph(self, graph_variables) -> nx.Graph:
        init_graph = super()._interpolate_graph(graph_variables)

        # do error-check
        if not all(node in init_graph for node in graph_variables):
            raise RuntimeError(
                "Not all nodes (observational and f-nodes) are part of the init graph."
            )
        return init_graph


def make_context(
    context: Optional[Context] = None, create_using=ContextBuilder
) -> Union[ContextBuilder, InterventionalContextBuilder]:
    """Create a new ContextBuilder instance.

    Returns
    -------
    result : ContextBuilder, InterventionalContextBuilder
        The new ContextBuilder instance

    Examples
    --------
    This creates a context object denoting that there are three observed
    variables, ``(1, 2, 3)``.
    >>> context_builder = make_context()
    >>> context = context_builder.variables([1, 2, 3]).build()

    Notes
    -----
    :class:`~.context.Context` objects are dataclasses that creates a dictionary-like access
    to causal context metadata. Copying relevant information from a Context
    object into a `ContextBuilder` is all supported with the exception of
    state variables. State variables are not copied over. To set state variables
    again, one must build the Context and then call
    :py:meth:`~.context.Context.state_variable`.
    """
    result = create_using()
    if context is not None:
        # we create a copy of the ContextBuilder with the current values
        # in the context
        ctx_params = context.get_params()
        for param, value in ctx_params.items():
            if getattr(result, param, None) is not None:
                getattr(result, param)(value)

    return result
