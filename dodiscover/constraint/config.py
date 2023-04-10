from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    """Meta class to enable easy checks for Enums."""

    def __contains__(cls, item):
        """Allows 'contain' checks.

        Example: ``is 'method' in EnumClass``.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class ConditioningSetSelection(Enum, metaclass=MetaEnum):
    """Available methods for selecting the conditioning sets when learning a skeleton.

    Given a pair of nodes in a graph, (X, Y), this enumeration selects a strategy
    for choosing conditioning sets to be checked for conditional independence.

    Notes
    -----
    Allows 'contains' checks because of the metaclass. For example,
    one can run ``"complete" in ConditioningSetSelection``, which would
    return `True`.
    """

    COMPLETE = "complete"
    """Considers all possible combinations of nodes in the graph that are
    not (X,Y).
    """

    NBRS = "neighbors"
    """Considers all current neighbors of (X,Y) in the graph.
    """

    NBRS_PATH = "neighbors_path"
    """Considers all neighbors of (X,Y) in the graph that are on a simple
    path between the two nodes.
    """

    PDS = "pds"
    """Considers all potentially d-separating sets for (X,Y) in the graph.
    This requires some initial collider information in the graph.
    """

    PDS_PATH = "pds_path"
    """Considers all PDS sets for (X,Y) in the graph that lie on a path
    between the two nodes.
    """
