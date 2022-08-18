"""
.. _ex-pc-algorithm:

====================================================================================
PC algorithm for causal discovery from observational data without latent confounders
====================================================================================

We will simulate some observational data from a Structural Causal Model (SCM) and
demonstrate how we will use the PC algorithm.

The PC algorithm works on observational data when there are no unobserved latent
confounders. That means for any observed set of variables, there is no common causes
that are unobserved. In other words, all exogenous variables then are assumed to be
independent.

.. currentmodule:: dodiscover
"""
# %%
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from pywhy_graphs import StructuralCausalModel
from pywhy_graphs.viz import draw
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover import PC, Context

# %%
# Simulate some data
# ------------------
# First we will simulate data, starting from a Structural Causal Model (SCM).
# This will then induce a causal graph, which we can visualize. Due to the
# Markov assumption, then we can use d-separation to examine which variables
# are conditionally independent.

# set a random seed to make example reproducible
seed = 12345
rng = np.random.RandomState(seed=seed)

# construct a causal graph that will result in
# x -> y <- z -> w
func_uz = lambda: rng.binomial(n=1, p=0.25)
func_ux = lambda: rng.binomial(n=1, p=0.4)
func_uy = lambda: rng.binomial(n=1, p=0.4)
func_uw = lambda: rng.binomial(n=1, p=0.5)
func_x = lambda u_x: 2 * u_x
func_y = lambda x, u_y, z: x * u_y + z
func_z = lambda u_z: u_z
func_w = lambda u_w, z: u_w + z

# construct the SCM and the corresponding causal graph
scm = StructuralCausalModel(
    exogenous={
        "u_x": func_ux,
        "u_y": func_uy,
        "u_z": func_uz,
        "u_w": func_uw,
    },
    endogenous={"x": func_x, "y": func_y, "z": func_z, "w": func_w},
)
G = scm.get_causal_graph()

# sample the incomplete observational data
data = scm.sample(n=5000, include_latents=False)

# note the graph shows a collider and will not show
# the unobserved confounder
draw(G)

# %%
# Instantiate some conditional independence tests
# -----------------------------------------------
# To run a constraint-based structure learning algorithm such
# as the PC algorithm, we need a way to test for conditional
# independence (CI) constraints. There are various ways we can evaluate
# the algorithm.
#
# If we are applying the algorithm on real data, we would want to
# use the CI test that best suits the data. Note that because of
# finite sample sizes, any CI test is bound to make some errors, which
# results in incorrect orientations and edges in the learned graph.
#
# If we are interested in evaluating how the structure learning algorithm
# works in an ideal setting, we can use an oracle, which is imbued with the
# ground-truth graph, which can query all the d-separation statements needed.
# This can help one determine in a simulation setting, what is the best case
# graph the PC algorithm can learn.
oracle = Oracle(G)
ci_estimator = GSquareCITest(data_type="discrete")

# %%
# Define the context
# ------------------
# In PyWhy, we introduce the :class:`Context` class, which should be a departure from
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
context = Context(data=data)

# Alternatively, one could say specify some fixed edges. 
# Note that when specifying fixed edges, the resulting graph that is
# learned is not necessarily a "pure CPDAG". In that, there are more
# constraints than just the conditional independences. Therefore, one
# should use caution when specifying fixed edges if they are interested
# in leveraging ID or estimation algorithms that assume the learned
# structure is a "pure CPDAG".
#
# .. code-block::Python
#   included_edges = nx.Graph([('x', 'y')])
#   context = Context(data=data, included_edges=included_edges)

# %%
# Run structure learning algorithm
# --------------------------------
# Now we are ready to run the PC algorithm. First, we will show the output of
# the oracle, which is the best case scenario the PC algorithm can learn given
# an infinite amount of data.

pc = PC(ci_estimator=oracle)
pc.fit(context)

# The resulting completely partially directed acyclic graph (CPDAG) that is learned
# is a "Markov equivalence class", which encodes all the conditional dependences that
# were learned from the data.
graph = pc.graph_

dot_graph = draw(graph)
dot_graph.render(outfile="cpdag.png", view=True)

# %%
# Now, we will show the output given a real CI test, which performs CI hypothesis testing
# to determine CI in the data. Due to finite data and the presence of noise, there is
# always a possibility that the CI test makes a mistake.
pc = PC(ci_estimator=ci_estimator)
pc.fit(context)

# The resulting completely partially directed acyclic graph (CPDAG) that is learned
# is a "Markov equivalence class", which encodes all the conditional dependences that
# were learned from the data.
graph = pc.graph_

dot_graph = draw(graph)
dot_graph.render(outfile="cpdag.png", view=True)
