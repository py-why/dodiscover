import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed
from pywhy_graphs.array.export import clearn_arr_to_graph
from scipy import stats

from dodiscover import FCI, make_context
from dodiscover.ci import FisherZCITest
from dodiscover.metrics import confusion_matrix_networks


def simulate_gcm():
    # set a random seed to make example reproducible
    seed = 12345
    rng = np.random.RandomState(seed=seed)
    # set a random seed to make example reproducible
    set_random_seed(1234)

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

    # construct a causal graph that will result in
    # x -> y <- z -> w
    edge_list = [
        ("x4", "x1"),
        ("x2", "x5"),
        ("x3", "x2"),
        ("x3", "x4"),
        ("x2", "x6"),
        ("x3", "x6"),
        ("x4", "x6"),
        ("x5", "x6"),
    ]
    latent_edge_list = [("x12", "x1"), ("x12", "x2"), ("x45", "x4"), ("x45", "x5")]
    G = nx.DiGraph(edge_list + latent_edge_list)

    causal_model = gcm.ProbabilisticCausalModel(G)

    for node in G.nodes:
        parents = set(G.predecessors(node))

        # root node
        random_mean = rng.normal()
        random_scale = rng.normal()
        gcm_distrib = gcm.ScipyDistribution(stats.norm, loc=random_mean, scale=random_scale)
        if len(parents) == 0:
            causal_model.set_causal_mechanism(node, gcm_distrib)
        else:
            random_coeff = rng.normal()

            causal_model.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    prediction_model=MyCustomModel(random_coeff), noise_model=gcm_distrib
                ),
            )

    # Fit here would not really fit parameters, since we don't do anything in the fit method.
    # Here, we only need this to ensure that each FCM has the correct local hash (i.e., we
    # get an inconsistency error if we would modify the graph afterwards without updating
    # the FCMs). Having an empty data set is a small workaround, since all models are
    # pre-defined.
    gcm.fit(causal_model, pd.DataFrame(columns=G.nodes))

    return causal_model


def test_fci_against_causallearn():
    n_samples = 5000
    alpha = 0.05
    causal_model = simulate_gcm()

    # sample the observational data
    data = gcm.draw_samples(causal_model, num_samples=n_samples)

    # drop latent confounders
    data.drop(["x45", "x12"], inplace=True)

    # setup input data for algorithms
    data_arr = data.to_numpy()
    context = make_context().variables(data=data).build()

    # First run causallearn
    clearn_graph, edges = fci(data_arr, independence_test_method="fisherz", alpha=alpha)

    # Second run FCI
    ci_test = FisherZCITest()
    fci_alg = FCI(ci_estimator=ci_test, alpha=alpha)
    fci_alg.fit(data, context)

    dodiscover_graph = fci_alg.graph_

    # first compare the adjacency structure
    clearn_graph = clearn_arr_to_graph(clearn_graph, arr_idx=data.columns, graph_type="pag")
    cm = confusion_matrix_networks(dodiscover_graph, clearn_graph)
