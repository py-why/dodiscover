"""
Wrapper for the GIN algorithm in causal-learn
TODO: Need type hints
"""
from causallearn.search.HiddenCausal.GIN.GIN import GIN as GIN_
import numpy.typing as npt
from pywhy_graphs import CPDAG

class GIN:
    """
    Dodiscover wrapper for GIN in causal-learn package
    """
    def __init__(self, indep_test_method="kci", alpha=0.05)-> None:
        """
        Using default parameters from GIN.
        - indep_test_method: str, optional (default='kci')
        The independence test method used by GIN.
        - alpha: float, optional (default=0.05)
        The significance level for the independence test.
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
        """Converts a causal-learn graph to a partially directed acyclic graph (CPDAG).
    
        Parameters:
            - cl_graph (causal_learn Graph): Any
            A causal-learn graph object.
    
        Returns:
            - pdag: CPDAG
            A CPDAG object.
        """
        def _extract_edgelists(adj_mat: npt, names:str)-> tuple:
            """Extracts directed and undirected edges from an adjacency matrix.
        
                Parameters:
                    - adj_mat: numpy array
                        The adjacency matrix of the graph.
                    - names: list of str
                        The names of the nodes in the graph.
            
                Returns:
                - directed_edges: list of tuples
                    The directed edges of the graph.
                - undirected_edges: list of sets
                    The undirected edges of the graph.
            """
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
