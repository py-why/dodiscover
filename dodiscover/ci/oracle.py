from typing import Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

from dodiscover.typing import Column

from .._protocol import GraphProtocol
from .base import BaseConditionalIndependenceTest


class Oracle(BaseConditionalIndependenceTest):
    """Oracle conditional independence testing.

    Used for unit testing and checking intuition.

    Parameters
    ----------
    graph : nx.DiGraph | GraphProtocol
        The ground-truth causal graph.
    """

    def __init__(self, graph: GraphProtocol) -> None:
        self.graph = graph

    def test(
        self,
        df: pd.DataFrame,
        x_var: Column,
        y_var: Column,
        z_covariates: Optional[Set[Column]] = None,
    ):
        """Conditional independence test given an oracle.

        Checks conditional independence between 'x_var' and 'y_var'
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
        x_var : node
            A node in the dataset.
        y_var : node
            A node in the dataset.
        z_covariates : set
            The set of variables to check that separates x_var and y_var.

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
        self._check_test_input(df, x_var, y_var, z_covariates)

        # just check for d-separation between x and y
        # given sep_set
        if isinstance(self.graph, nx.DiGraph):
            is_sep = nx.d_separated(self.graph, {x_var}, {y_var}, z_covariates)
        else:
            from graphs import m_separated

            is_sep = m_separated(self.graph, {x_var}, {y_var}, z_covariates)

        if is_sep:
            pvalue = 1
            test_stat = 0
        else:
            pvalue = 0
            test_stat = np.inf
        return test_stat, pvalue
