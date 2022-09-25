from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class SkeletonMethods(Enum, metaclass=MetaEnum):
    """Available methods for learning a skeleton from data."""

    COMPLETE = "complete"
    NBRS = "neighbors"
    NBRS_PATH = "neighbors_path"
    PDS = "pds"
    PDS_PATH = "pds_path"