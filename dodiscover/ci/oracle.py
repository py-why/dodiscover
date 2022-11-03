from typing import Optional, Set, Union

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover.typing import Column

from .._protocol import Graph, TimeSeriesGraph
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

    def __init__(self, graph: Union[Graph, TimeSeriesGraph]) -> None:
        self.graph = graph

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

        # just check for d-separation between x and y given sep_set
        if isinstance(self.graph, nx.DiGraph):
            is_sep = nx.d_separated(self.graph, x_vars, y_vars, z_covariates)
        else:
            is_sep = nx.m_separated(self.graph, x_vars, y_vars, z_covariates)

        if is_sep:
            pvalue = 1
            test_stat = 0
        else:
            pvalue = 0
            test_stat = np.inf
        return test_stat, pvalue
