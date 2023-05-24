from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Set, Tuple
from warnings import warn

import networkx as nx

from ._protocol import Graph
from .base import BasePyWhy
from .typing import Column


# TODO: we should try to make the thing frozen
# - this would require easy copying of the Context into a new context
# - but resetting e.g. only say one variable like the init_graph
# - IDEAS: perhaps add a function `new_context = copy_context(context, **kwargs)`
# - where kwargs are the things to change.
@dataclass(
    eq=True,
    # frozen=True
)
class Context(BasePyWhy):
    """Context of assumptions, domain knowledge and data.

    This should NOT be instantiated directly. One should instead
    use `dodiscover.make_context` to build a Context data structure.

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
    state_variables : Dict
        Name of intermediate state variables during the learning process.
    intervention_targets : list of tuple
        List of intervention targets (known, or unknown), which correspond to
        the nodes in the graph (known), or indices of datasets that contain
        interventions (unknown).

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

    **Testing for equality**

    Currently, testing for equality is done on all attributes that are not
    graphs. Defining equality among graphs is ill-defined, and as such, we
    leave testing of the internal graphs to users. Some checks of equality
    for example can be :func:`networkx.algorithms.isomorphism.is_isomorphic`
    for checking isomorphism among two graphs.
    """

    observed_variables: Set[Column]
    latent_variables: Set[Column]
    state_variables: Dict[str, Any]
    init_graph: Graph = field(compare=False)
    included_edges: nx.Graph = field(compare=False)
    excluded_edges: nx.Graph = field(compare=False)

    ########################################################
    # for interventional data
    ########################################################
    # the number of distributions we expect to have access to
    num_distributions: int = field(default=1)

    # whether or not observational distribution is present
    obs_distribution: bool = field(default=True)

    # (optional) known intervention targets, corresponding to nodes in the graph
    intervention_targets: List[Tuple[Column]] = field(default_factory=list)

    # (optional) mapping F-nodes to their symmetric difference intervention targets
    symmetric_diff_map: Dict[Any, FrozenSet] = field(default_factory=dict)

    # sigma-map mapping F-nodes to their distribution indices
    sigma_map: Dict[Any, Tuple] = field(default_factory=dict)
    f_nodes: List = field(default_factory=list)

    ########################################################
    # for general multi-domain data
    ########################################################
    # the number of domains we expect to have access to
    num_domains: int = field(default=1)

    # map each augmented node to a tuple of domains (e.g. (0, 1), or (1,))
    domain_map: Dict[Any, Tuple] = field(default_factory=dict)
    s_nodes: List = field(default_factory=list)

    def add_state_variable(self, name: str, var: Any) -> "Context":
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
        self.state_variables[name] = var
        return self

    def state_variable(self, name: str, on_missing: str = "raise") -> Any:
        """Get a state variable.

        Parameters
        ----------
        name : str
            The name of the state variable.
        on_missing : {'raise', 'warn', 'ignore'}
            Behavior if ``name`` is not in the dictionary of state variables.
            If 'raise' (default) will raise a RuntimeError. If 'warn', will
            raise a UserWarning. If 'ignore', will return `None`.

        Returns
        -------
        state_var : Any
            The state variable.
        """
        if name not in self.state_variables and on_missing != "ignore":
            err_msg = f"{name} is not a state variable: {self.state_variables}"
            if on_missing == "raise":
                raise RuntimeError(err_msg)
            elif on_missing == "warn":
                warn(err_msg)

        return self.state_variables.get(name)

    def copy(self) -> "Context":
        """Create a deepcopy of the context."""
        return Context(**self.get_params(deep=True))

    ###############################################################
    # Methods for interventional data.
    ###############################################################
    def get_non_augmented_nodes(self) -> Set:
        """Get the set of non f-nodes."""
        non_augmented_nodes = set()
        f_nodes = set(self.f_nodes)
        s_nodes = set(self.s_nodes)
        for node in self.init_graph.nodes:
            if node not in f_nodes and node not in s_nodes:
                non_augmented_nodes.add(node)
        return non_augmented_nodes

    def get_augmented_nodes(self) -> Set:
        """Get the set of f-nodes."""
        return set(self.f_nodes).union(set(self.s_nodes))

    def reverse_sigma_map(self) -> Dict:
        """Get the reverse sigma-map."""
        reverse_map = dict()
        for node, mapping in self.sigma_map.items():
            reverse_map[mapping] = node
        return reverse_map
