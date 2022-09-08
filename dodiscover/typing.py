from typing import Dict, List, Set, Union

# Pandas DataFrame columns that are also compatible with Graph nodes
Column = Union[int, float, str]

# The separating set used in constraint-based causal discovery
SeparatingSet = Dict[Column, Dict[Column, List[Set[Column]]]]
