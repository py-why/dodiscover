from typing import Any, Dict, List, Set

import pandas as pd

from dodiscover import Graph


def dummy_sample(G: Graph):
    """Sample an empty dataframe with columns as the nodes.

    Used for oracle testing.
    """
    return pd.DataFrame({column: [] for column in G.nodes})  # type: ignore


def is_in_sep_set(
    check_var, sep_set: Dict[str, Dict[str, List[Set[Any]]]], x_var, y_var, mode: str = "any"
):
    """Check that a variable is not in any separating set between 'X' and 'Y'.

    Parameters
    ----------
    check_var : node
        The variable to check.
    sep_set : Dict[str, Dict[str, List[Set[Any]]]]
        The separating sets between any two variables 'X' and 'Y'.
    x_var : node
        The 'X' variable.
    y_var : node
        The 'Y' variable.
    mode : str
        Whether to check in 'any' separating sets, or check if it is in 'all' separating sets,
        or if it is in 'some' separating sets, but not all. Default is 'any'.

    Returns
    -------
    bool
        Whether or not 'check_var' is in all separating sets of 'x_var' and 'y_var'.
    """
    if mode == "any":
        func = any
    elif mode == "all":
        func = all
    elif mode == "some":
        return any(check_var in _sep_set for _sep_set in sep_set[x_var][y_var]) and not all(
            check_var in _sep_set for _sep_set in sep_set[x_var][y_var]
        )
    return func(check_var in _sep_set for _sep_set in sep_set[x_var][y_var])
