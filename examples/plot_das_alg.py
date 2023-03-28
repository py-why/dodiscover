"""
.. _ex-das-algorithm:

=======================================================================================
DAS algorithm for causal discovery from observational data without latent confounders
=======================================================================================

We will simulate some observational data from a Structural Causal Model (SCM) and
demonstrate how we will use the DAS algorithm.

The DAS algorithm works on observational data when there are no unobserved latent
confounders. That means for any observed set of variables, there is no common causes
that are unobserved. In other words, all exogenous variables then are assumed to be
independent. Additionally, it works under the assumptions of Additive Noise Model with
nonlinear mechanisms and Gaussian noise terms.

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
from dodiscover.toporder.das import DAS
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
# Now we are ready to run the DAS algorithm. The methods performs inference
# in two phases. First it estimates the topological order of the nodes in the
# graphs. This passage is identical to SCORE. The topological order is found
# iteratively according to the following procedure
# 1. DAS estimates the Hessian of log p(x), with p(x) joint distribution of
# nodes in the graph
# 2. Let H := Hessian(log p(x)). DAS selects a leaf in the graph by finding
# the diagonal term of H with minimum variance,
# i.e. by computing "argmin Var[diagonal(H)]"
# 3. DAS remove the leaf in from the graph,  and repeats from 1. to 3.
# iteratively up to the source nodes.
# The main difference between DAS and SCORE is in the pruning of the fully
# connected graph representation of the topological ordering.
# DAS removes edges by inspection of the off diagonal entries of the Hessian
# of log p(x): in particular, to verify existence of an edge i -> j admitted
# by the ordering, DAS performs hypothesis testing on the [i, j] entry of the
# Hessian, the null hypothesis being that the mean of H[i, j] is zero.
# If we fail to reject the null, we remove the edge i -> j from the fully
# connected matrix. Finally, an additional pruning step via sparse regression
# as for SCORE is performed: given the preliminary edge removeal by inspection
# of the off-diagonal elements of the Hessian, this pruning step has constant
# execution time, making DAS overall pruning procedure one order fo magnitude
# quicker than SCORE in the nuber of nodes.
# To clarify: let "d" be th enumber of nodes in the graph, "n" the number of
# samples in the dataset:
# - DAS pruning runs in O(d^2) time
# - SCORE pruning runs in O(nd^3) time
# The computational advantage of DAS is manifest in large graphs.

das = DAS()
das.fit(data, context)

# DAS estimates a directed acyclic graph (DAG) and the topological order
# of the nodes in the graph. DAS is consistent in the infinite samples
# limit, meaning that it might return faulty estimates due to the finiteness
# of the data.
graph = das.graph_
order = das.order_

# "das_full_dag.png" visualizes the fully connected DAG representation of
# the inferred topological ordering "order".
# "das_dag.png" visualizes the fully connected DAG after pruning with
# sparse regression.
dot_graph = draw(nx.from_numpy_array(full_DAG(order), create_using=nx.DiGraph))
dot_graph.render(outfile="das_full_dag.png", view=True)

dot_graph = draw(graph)
dot_graph.render(outfile="das_dag.png", view=True)

# %%
#
