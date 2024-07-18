"""
Wrapper for the GIN latent variable causal discovery algorithm in causal-learn
"""

from typing import Optional

from pandas import DataFrame
from pywhy_graphs.export import clearn_to_graph


class GIN:
    """Wrapper for GIN in the causal-learn package.

    The GIN algorithm is a causal discovery algorithm that learns latent
    variable structure.  We can view it as a causal representation learning
    algorithm.

    Given an observed set of variables, the GIN algorithm tries to learn a set
    of latent parents that d-separate subsets of the observed variables.  GIN
    will also learn undirected structure between the latent parents.

    In GIN, the latent variables are always assumed to be parents of the
    observed. Further, it will not learn direct causal edges between
    the observed variables. In that sense, we can view it as a causal
    representation learning algorithm that learns latent high-level variables
    and structure between them from low-level observed variables.

    The GIN algorithm assumes a linear non-Gaussian latent variable model
    of the observed variables given the latent variables.  One should not
    expect it to work if the true relationship is Gaussian.

    GIN stands for "generalized independent noise" (GIN) condition. Roughly,
    the GIN condition is used to divide observed variables into subsets that
    are d-separated given the latent variables.

    See :footcite:`xie2020generalized` and :footcite:`dai2022independence`
    for full details on the algorithm.See https://causal-learn.readthedocs.io
    for the causal-learn documentation.

    Parameters
    ----------
    indep_test_method : str
        The method to use for testing independence.  The default argument is
        "kci" for kernel conditional independence testing. Another option is
        "hsic" for the Hilbert Schmidt Independence Criterion. This is a
        wrapper for causal-learn's GIN implementation and the causal-learn devs
        may or may not add other options in the future.
    alpha : float
        The significance level for independence tests, by default 0.05

    Attributes
    ----------
    graph_ : CPDAG
        The estimated causal graph.
    causal_learn_graph_ : CausalGraph
        The causal graph object from causal-learn. Internally, we convert this
        to a  network-like graph object that supports CPDAGs. This is stored in
         the ``graph_`` fitted attribute.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, ci_estimator_method: str = "kci", alpha: float = 0.05):
        """Initialize GIN object with specified parameters."""

        self.graph = None

        # GIN default parameters.
        self.ci_estimator_method = ci_estimator_method
        self.alpha = alpha
        # The follow objects are specific to causal-learn, perhaps they should
        # go in a base class too.
        self.causal_learn_graph_ = None

    def learn_graph(self, data: DataFrame, context: Optional[DataFrame] = None):
        """Fit the GIN model to data.
        Currently the context object is not used.

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
        """
        if context is None:
            # make a private Context object to store causal context used in this algorithm
            # store the context
            from dodiscover.context_builder import make_context

            context = make_context().build()

        from causallearn.search.HiddenCausal.GIN.GIN import GIN as GIN_

        causal_learn_graph, _ = GIN_(data.to_numpy(), self.ci_estimator_method, self.alpha)
        self.causal_learn_graph_ = causal_learn_graph
        names = [n.name for n in causal_learn_graph.nodes]
        adj_mat = causal_learn_graph.graph
        self.graph = clearn_to_graph(adj_mat, names, "cpdag")
