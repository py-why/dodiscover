"""Tests wrapper for GIN algorithm from causal"""

import networkx as nx
import numpy as np
import pandas as pd
from pywhy_graphs import CPDAG

from dodiscover import make_context
from dodiscover.replearning.gin import GIN


def test_estimate_gin_testdata():
    """Test the wrapper to the causal-learn GIN algorithm for estimating the causal DAG."""

    # Sim data
    np.random.seed(123)
    num_samples = 1000
    # First latent is a uniform
    latent_var_1 = np.random.uniform(0, 100, num_samples)
    # Second latent is caused by first via nonlinear transform
    latent_var_2 = np.array(list(map(lambda u: 100 * 0.03 * u / (1 + 0.03 * u), latent_var_1)))
    # Observed variables. X1 and X2 are caused by L1, X3 and X4 are caused by L2
    observed_vars = np.array(
        [
            latent_var_1 + np.random.normal(0, 1, num_samples),  # X1 caused by L1
            (100 - latent_var_1)
            + np.random.normal(0, 1, num_samples),  # X2 caused by L1, mirros X1
            latent_var_2 + np.random.normal(0, 1, num_samples),  # X3 caused by L2
            (100 - latent_var_2)
            + np.random.normal(0, 1, num_samples),  # X4 caused by L2, mirrors X3
        ]
    ).transpose()
    data = pd.DataFrame(observed_vars, columns=["X1", "X2", "X3", "X4"])

    g_answer = CPDAG(
        [
            ("L1", "X1"),
            ("L1", "X2"),
            ("L2", "X3"),
            ("L2", "X4"),
        ],
        [
            ("L1", "L2"),
        ],
    )

    context = make_context().variables(data=data).build()
    gin = GIN()
    gin.learn_graph(data, context)
    pdag = gin.graph

    assert nx.is_isomorphic(pdag.sub_undirected_graph(), g_answer.sub_undirected_graph())
    assert nx.is_isomorphic(pdag.sub_directed_graph(), g_answer.sub_directed_graph())
