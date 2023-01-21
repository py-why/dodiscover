"""Tests wrapper for GIN algorithm from causal"""
import networkx as nx
import numpy as np
import pandas as pd

from dodiscover import make_context
from dodiscover.scm.deeplearning.causica_ import Causica


def test_estimate_causica_testdata():
    """
    Test the wrapper to the causal-learn Causica algorithm for estimating
    the causal DAG.
    """
    # Sim data
    np.random.seed(123)
    num_samples = 30
    X1 = np.random.normal(0, 1, size=num_samples)
    X2 = np.random.normal(0, 1, size=num_samples)
    noise = np.random.normal(0, 1, size=num_samples)
    Y = X1 + X2 + X1 * X1 + X2 * X2 + X1 * X2 + noise
    data = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})
    g_answer = nx.DiGraph([("X1", "Y"), ("X2", "Y")])

    context = make_context().variables(data=data).build()
    # These parameters were not given much thought, other than making the test
    # run quickly.
    model_params = {
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "imputation": False,
        "lambda_dag": 10.0,
        "lambda_sparse": 1.0,
        "lambda_prior": 0.0,
        "tau_gumbel": 0.25,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
    }
    causica = Causica(model_params)

    training_params = {
        "learning_rate": 0.1,
        "batch_size": num_samples,
        "standardize_data_mean": True,
        "standardize_data_std": True,
        "rho": 1.0,
        "safety_rho": 1e18,
        "alpha": 0.0,
        "safety_alpha": 1e18,
        "tol_dag": 1e-9,
        "progress_rate": 0.65,
        "max_steps_auglag": 10,
        "max_auglag_inner_epochs": 200,
        "max_p_train_dropout": 0.0,
        "reconstruction_loss_factor": 1.0,
        "anneal_entropy": "noanneal",
    }
    causica.fit(data, context, training_params)
    dag = causica.graph

    assert nx.is_isomorphic(dag, g_answer)
