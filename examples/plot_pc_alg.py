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
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from pywhy_graphs import StructuralCausalModel
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover import PC

# %%
# Simulate some data

# set a random seed to make example reproducible
seed = 12345
rng = np.random.RandomState(seed=seed)

# construct a causal graph that will result in
# x -> y <- z
func_uz = lambda: rng.binomial(n=1, p=0.25)
func_uxy = lambda: rng.binomial(n=1, p=0.4)
func_x = lambda u_xy: 2 * u_xy
func_y = lambda x, u_xy, z: x * u_xy + z
func_z = lambda u_z: u_z

# construct the SCM and the corresponding causal graph
scm = StructuralCausalModel(
    exogenous={
        "u_xy": func_uxy,
        "u_z": func_uz,
    },
    endogenous={"x": func_x, "y": func_y, "z": func_z},
)
G = scm.get_causal_graph()

# sample the incomplete observational data
data = scm.sample(n=5000, include_latents=False)

# note the graph shows a collider and will not show
# the unobserved confounder
G.draw()

# %%
# Instantiate some conditional independence tests
oracle = Oracle(G)
ci_estimator = GSquareCITest(data_type="discrete")

# %%
# Now we are ready to run the FCI algorithm.

fci = FCI(ci_estimator=ci_estimator)
fci.fit(data)

# the resulting partial ancestral graph (PAG) that is learned
graph = fci.graph_
graph.draw()
