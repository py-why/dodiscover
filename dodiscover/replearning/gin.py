"""
Wrapper for the GIN algorithm in causal-learn
TODO: Need type hints
"""
from causallearn.search.HiddenCausal.GIN.GIN import GIN as GIN_
from pywhy_graphs import CPDAG

class GIN:
    """Wrapper for GIN in the causal-learn package.

    Parameters
    ----------
    indep_test_method : str
        The method to use for testing independence, by default "kci"
    alpha : float
        The significance level for independence tests, by default 0.05

    Attributes
    ----------
    graph_ : CPDAG
        The estimated causal graph.
    causal_learn_graph : CausalGraph
        The causal graph object from causal-learn.
    causal_ordering : list of str
        The causal ordering of the variables.
    """
    def __init__(self, indep_test_method: str="kci", alpha: float=0.05):
        """Initialize GIN object with specified parameters."""

        self.graph_ = None # Should be in a base class

        # GIN default parameters.
        self.indep_test_method = indep_test_method
        self.alpha = alpha
        # The follow objects are specific to causal-learn, perhaps they should
        # go in a base class too.
        self.causal_learn_graph = None
        self.causal_ordering = None

    def _causal_learn_to_pdag(self, cl_graph):
        """Convert a causal-learn graph to a CPDAG object.
        
        Parameters
        ----------
        cl_graph : CausalGraph
            The causal-learn graph to be converted.
        
        Returns
        -------
        pdag : CPDAG
            The equivalent CPDAG object.
        """
        def _extract_edgelists(adj_mat, names):
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


    def fit(self, data: 'DataFrame', context: 'DataFrame'):
        """Fit to data.
        
        Parameters
        ----------
        data : DataFrame
            The data to fit to.
        context : DataFrame
            The context variables to use as constraints.
        
        Returns
        -------
        self : GIN
            The fitted GIN object.

        TODO: How to apply context constraints?  Need to create issue"""
        causal_learn_graph, ordering = GIN_(
            data.to_numpy(),
            self.indep_test_method,
            self.alpha
        )
        self.causal_learn_graph = causal_learn_graph
        self.causal_ordering = ordering
        self.graph_ = self._causal_learn_to_pdag(causal_learn_graph)
