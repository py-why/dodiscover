from typing import Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover.typing import Column

from .._protocol import Graph
from .base import BaseConditionalIndependenceTest


class Oracle(BaseConditionalIndependenceTest):
    """Oracle conditional independence testing.

    Used for unit testing and checking intuition.

    Parameters
    ----------
    graph : nx.DiGraph | Graph
        The ground-truth causal graph.
    """

    _allow_multivariate_input: bool = True

    def __init__(self, graph: Graph, included_nodes: Set[Column] = None) -> None:
        self.graph = graph
        self.included_nodes = included_nodes

    def test(
        self,
        df: pd.DataFrame,
        x_vars: Set[Column],
        y_vars: Set[Column],
        z_covariates: Optional[Set[Column]] = None,
    ):
        """Conditional independence test given an oracle.

        Checks conditional independence between 'x_vars' and 'y_vars'
        given 'z_covariates' of variables using the causal graph
        as an oracle. The oracle uses d-separation statements given
        the graph to query conditional independences. This is known
        as the Markov property for graphs
        :footcite:`Pearl_causality_2009,Spirtes1993`.

        Parameters
        ----------
        df : pd.DataFrame of shape (n_samples, n_variables)
            The data matrix. Passed in for API consistency, but not
            used.
        x_vars : node
            A node in the dataset.
        y_vars : node
            A node in the dataset.
        z_covariates : set
            The set of variables to check that separates x_vars and y_vars.

        Returns
        -------
        statistic : None
            A return argument for the statistic.
        pvalue : float
            The pvalue. Return '1.0' if not independent and '0.0'
            if they are.

        References
        ----------
        .. footbibliography::
        """
        self._check_test_input(df, x_vars, y_vars, z_covariates)

        # generate a set of included nodes always in the Z-covariates
        if self.included_nodes is None:
            included_nodes = set()
        else:
            included_nodes = (
                set(self.included_nodes).difference(set(x_vars)).difference(set(y_vars))
            )
        z_covariates = set(z_covariates).union(included_nodes)

        # just check for d-separation between x and y given sep_set
        if isinstance(self.graph, nx.DiGraph):
            is_sep = nx.d_separated(self.graph, x_vars, y_vars, z_covariates)
        else:
            import pywhy_graphs.networkx as pywhy_nx

            is_sep = pywhy_nx.m_separated(self.graph, x_vars, y_vars, z_covariates)

        if is_sep:
            pvalue = 1
            test_stat = 0
        else:
            pvalue = 0
            test_stat = np.inf
        return test_stat, pvalue
