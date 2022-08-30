from typing import Dict, List, Set, Union

# Pandas DataFrame columns that are also compatible with Graph nodes
Column = Union[int, float, str]
SeparatingSet = Dict[Column, Dict[Column, List[Set[Column]]]]
