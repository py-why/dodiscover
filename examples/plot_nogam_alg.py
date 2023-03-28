"""
.. _ex-nogam-algorithm:

=======================================================================================
NoGAM algorithm for causal discovery from observational data without latent confounders
=======================================================================================

We will simulate some observational data from a Structural Causal Model (SCM) and
demonstrate how we will use the NoGAM algorithm.

The NoGAM algorithm works on observational data when there are no unobserved latent
confounders. That means for any observed set of variables, there is no common causes
that are unobserved. In other words, all exogenous variables then are assumed to be
independent. Additionally, it works under the assumptions of Additive Noise Model.

.. currentmodule:: dodiscover
"""
# %%
# Authors: Francesco Montagna <francesco.montagna997@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import networkx as nx
from scipy import stats
from pywhy_graphs.viz import draw
from dodiscover import make_context
from dodiscover.toporder.nogam import NoGAM
from dodiscover.toporder.utils import full_DAG
import pandas as pd
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed


# %%
# Simulate some data
# ------------------
# First we will simulate data, starting from an Additive Noise Model (ANM).
# This will then induce a causal graph, which we can visualize. Due to the
# Markov assumption, then we can use d-separation to examine which variables
# are conditionally independent.

# set a random seed to make example reproducible
seed = 12345
rng = np.random.RandomState(seed=seed)


class MyCustomModel(gcm.PredictionModel):
    def __init__(self, coefficient):
        self.coefficient = coefficient

    def fit(self, X, Y):
        # Nothing to fit here, since we know the ground truth.
        pass

    def predict(self, X):
        return self.coefficient * X

    def clone(self):
        # We don't really need this actually.
        return MyCustomModel(self.coefficient)


# set a random seed to make example reproducible
set_random_seed(1234)

# construct a causal graph that will result in
# x -> y <- z -> w
G = nx.DiGraph([("x", "y"), ("z", "y"), ("z", "w")])

causal_model = gcm.ProbabilisticCausalModel(G)
causal_model.set_causal_mechanism("x", gcm.ScipyDistribution(stats.binom, p=0.5, n=1))
causal_model.set_causal_mechanism("z", gcm.ScipyDistribution(stats.binom, p=0.9, n=1))
causal_model.set_causal_mechanism(
    "y",
    gcm.AdditiveNoiseModel(
        prediction_model=MyCustomModel(1),
        noise_model=gcm.ScipyDistribution(stats.binom, p=0.8, n=1),
    ),
)
causal_model.set_causal_mechanism(
    "w",
    gcm.AdditiveNoiseModel(
        prediction_model=MyCustomModel(1),
        noise_model=gcm.ScipyDistribution(stats.binom, p=0.5, n=1),
    ),
)

# Fit here would not really fit parameters, since we don't do anything in the fit method.
# Here, we only need this to ensure that each FCM has the correct local hash (i.e., we
# get an inconsistency error if we would modify the graph afterwards without updating
# the FCMs). Having an empty data set is a small workaround, since all models are
# pre-defined.
gcm.fit(causal_model, pd.DataFrame(columns=["x", "y", "z", "w"]))

# sample the observational data
data = gcm.draw_samples(causal_model, num_samples=500)

print(data.head())
print(pd.Series({col: data[col].unique() for col in data}))
draw(G)

# %%
# Define the context
# ------------------
# In PyWhy, we introduce the :class:`context.Context` class, which should be a departure from
# "data-first causal discovery," where users provide data as the primary input to a
# discovery algorithm. This problem with this approach is that it encourages novice
# users to see the algorithm as a philosopher's stone that converts data to causal
# relationships. With this mindset, users tend to surrender the task of providing
# domain-specific assumptions that enable identifiability to the algorithm. In
# contrast, PyWhy's key strength is how it guides users to specifying domain
# assumptions up front (in the form of a DAG) before the data is added, and
# then addresses identifiability given those assumptions and data. In this sense,
# the Context class houses both data, apriori assumptions and other relevant data
# that may be used in downstream structure learning algorithms.
context = make_context().variables(data=data).build()

# Alternatively, one could say specify some fixed edges.
# .. code-block::Python
#   included_edges = nx.Graph([('x', 'y')])
#   context = make_context().edges(include=included_edges).build()

# %%
# Run structure learning algorithm
# --------------------------------
# Now we are ready to run the NoGAM algorithm. The methods performs inference
# in two phases. First it estimates the topological order of the nodes in the
# graphs. This is done iteratively according to the following procedure
# 1. NoGAM estimate the gradient of log p(x), i.e. the score function, via
# Stein gradient estimator.
# 2. For each node in the graph, NoGAM estimates its residual by kernel ridge
# regression (or any nonlinear regressor): this is done by regressing
# each variable 'j' against all the remaining variables 1, .., j-1, j+1, ..
# Exact estimation of the residuals is possible if and only if j is a leaf.
# 3. For each node 'j', NoGAM estimates the j-th entry of the score function
# using the residual of the j-th variable estimates in step 2. as a predictor.
# The mean squared error (MSE) of this regression problem is zero if and only
# if 'j' is a leaf node.
# 4. NoGAM removes the identified leaf from the graph, and repeats steps 1 to 4.
# up to the source nodes.
# Given the inferred topological order, NoGAM prunes the graph by with all
# edges admitted by such ordering, by doing sparse regression to choose the
# relevant variables. Variable selection is done by thresholding on the
# p-values of the coefficients associated to the potential parents of a node.
# For instance, consider a graph G with 3 vertices V = {1, 2, 3}. For
# simplicity let the topological order be trivial, i.e. {1, 2, 3}. The unique
# fully connected adjacency matrix compatible with such ordering is the upper
# triangular matrix np.triu(np.ones((3, 3)), k=1) with all ones above the
# diagonal.
# The difference between SCORE and NoGAM is that inference of the topological
# order with NoGAM does not require any assumption on the distribution of the
# noise terms in the ANM.
nogam = NoGAM()
nogam.fit(data, context)

# NoGAM estimates a directed acyclic graph (DAG) and the topological order
# of the nodes in the graph. NoGAM is consistent in the infinite samples
# limit, meaning that it might return faulty estimates due to the finiteness
# of the data.
graph = nogam.graph_
order = nogam.order_

# "nogam_full_dag.png" visualizes the fully connected DAG representation of
# the inferred topological ordering "order".
# "nogam_dag.png" visualizes the fully connected DAG after pruning with
# sparse regression.
dot_graph = draw(nx.from_numpy_array(full_DAG(order), create_using=nx.DiGraph))
dot_graph.render(outfile="nogam_full_dag.png", view=True)

dot_graph = draw(graph)
dot_graph.render(outfile="nogam_dag.png", view=True)

# %%
#
