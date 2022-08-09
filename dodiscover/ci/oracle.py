from typing import Union

import numpy as np

from causal_networkx import ADMG, DAG
from causal_networkx.algorithms.d_separation import d_separated

from .base import BaseConditionalIndependenceTest


class Oracle(BaseConditionalIndependenceTest):
    """Oracle conditional independence testing.

    Used for unit testing and checking intuition.

    Parameters
    ----------
    graph : DAG | ADMG
        The ground-truth causal graph.
    """

    def __init__(self, graph: Union[ADMG, DAG]) -> None:
        self.graph = graph

    def test(self, df, x_var, y_var, z_covariates):
        """Conditional independence test given an oracle.

        Checks conditional independence between 'x_var' and 'y_var'
        given 'z_covariates' of variables using the causal graph
        as an oracle.

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
        """
        self._check_test_input(df, x_var, y_var, z_covariates)

        # just check for d-separation between x and y
        # given sep_set
        is_sep = d_separated(self.graph, x_var, y_var, z_covariates)

        if is_sep:
            pvalue = 1
            test_stat = 0
        else:
            pvalue = 0
            test_stat = np.inf
        return test_stat, pvalue


class ParentChildOracle(Oracle):
    """Parent and children oracle for conditional independence testing.

    An oracle that knows the definite parents and children of every node.
    """

    def get_children(self, x):
        """Return the definite children of node 'x'."""
        return self.graph.successors(x)

    def get_parents(self, x):
        """Return the definite parents of node 'x'."""
        return self.graph.predecessors(x)


class MarkovBlanketOracle(ParentChildOracle):
    """MB oracle for conditional independence testing.

    An oracle that knows the definite Markov Blanket of every node.
    """

    def __init__(self, graph: Union[ADMG, DAG]) -> None:
        super().__init__(graph)

    def get_markov_blanket(self, x):
        """Return the markov blanket of node 'x'."""
        return self.graph.markov_blanket_of(x)


class AncestralOracle(ParentChildOracle):
    """Oracle with access to ancestors of any specific node."""

    def get_ancestors(self, x):
        return self.graph.ancestors(x)
