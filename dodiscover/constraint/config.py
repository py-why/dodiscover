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


class SkeletonMethods(Enum, metaclass=MetaEnum):
    """Available methods for learning a skeleton from data.

    Notes
    -----
    Allows 'contains' checks because of the metaclass. For example,
    one can do ``is 'complete' in SkeletonMethods``, which would
    return True.
    """

    COMPLETE = "complete"
    NBRS = "neighbors"
    NBRS_PATH = "neighbors_path"
    PDS = "pds"
    PDS_PATH = "pds_path"
    PDS_T = "pds_t"
    PDS_T_PATH = 'pds_t_path'
