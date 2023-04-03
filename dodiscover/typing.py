from typing import Dict, List, Set, Union

import networkx as nx

# from .context import Context as BaseContext
# from .context_builder import ContextBuilder as BaseContextBuilder

# Pandas DataFrame columns that are also compatible with Graph nodes
Column = Union[int, float, str]

# The separating set used in constraint-based causal discovery
SeparatingSet = Dict[Column, Dict[Column, List[Set[Column]]]]

# The relevant networkx graphs accepted in this module
NetworkxGraph = Union[nx.Graph, nx.DiGraph]
