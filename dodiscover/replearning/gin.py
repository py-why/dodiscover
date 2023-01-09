"""
Wrapper for the GIN algorithm in causal-learn
TODO: Need type hints
"""
from causallearn.search.HiddenCausal.GIN.GIN import GIN as GIN_

from pywhy_graphs import CPDAG

class GIN:
    """
    Dodiscover wrapper for GIN in causal-learn package
    """
    def __init__(self, indep_test_method="kci", alpha=0.05):
        """
        Using default parameters from GIN.
        TODO: Add full set of GIN parameters with default
        TODO: Add a base class
        """
        self.graph_ = None # Should be in a base class

        # GIN default parameters.
        self.indep_test_method = indep_test_method
        self.alpha = alpha
        # The follow objects are specific to causal-learn, perhaps they should
        # go in a base class too.
        self.causal_learn_graph = None
        self.causal_ordering = None

    def _causal_learn_to_pdag(self, cl_graph):
        """"""
        def _extract_edgelists(adj_mat, names):
            directed_edges = []
            undirected_edges = []
            for i, row in enumerate(adj_mat):
                for j, item in enumerate(row):
                    if item != 0.:
                        if item == -1. and adj_mat[j][i] == -1.:
                            undirected_edges.append(set((names[j], names[i])))
                        if item == 1.:
                            directed_edges.append((names[j], names[i]))
            undirected_edges = list(set(tuple(edge) for edge in undirected_edges))
            return directed_edges, undirected_edges

        names = [n.name for n in cl_graph.nodes]
        adj_mat = cl_graph.graph
        directed_edges, undirected_edges = _extract_edgelists(
            adj_mat,
            names
        )
        pdag = CPDAG(directed_edges, undirected_edges)
        return pdag


    def fit(self, data, context):
        """Fit to data.
        TODO: How to apply context constraints?  Need to create issue"""
        causal_learn_graph, ordering = GIN_(
            data.to_numpy(),
            self.indep_test_method,
            self.alpha
        )
        self.causal_learn_graph = causal_learn_graph
        self.causal_ordering = ordering
        self.graph_ = self._causal_learn_to_pdag(causal_learn_graph)
