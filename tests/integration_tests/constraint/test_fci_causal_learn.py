import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed
from pywhy_graphs.export import clearn_to_graph
from scipy import stats

from dodiscover import FCI, make_context
from dodiscover.ci import FisherZCITest, GSquareCITest, Oracle
from dodiscover.metrics import confusion_matrix_networks, structure_hamming_dist


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
        random_scale = rng.uniform()
        gcm_distrib = gcm.ScipyDistribution(stats.norm, loc=random_mean, scale=random_scale)

        p = 0.5
        gcm_distrib = gcm.ScipyDistribution(stats.binom, n=1, p=p)

        weight_model = rng.normal()
        weight_model = rng.binomial(n=1, p=p)

        if len(parents) == 0:
            causal_model.set_causal_mechanism(node, gcm_distrib)
        else:
            causal_model.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    prediction_model=MyCustomModel(weight_model), noise_model=gcm_distrib
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
    n_samples = 2000
    alpha = 0.05
    causal_model = simulate_gcm()

    # sample the observational data
    data = gcm.draw_samples(causal_model, num_samples=n_samples)

    # drop latent confounders
    data.drop(["x45", "x12"], axis=1, inplace=True)

    # setup input data for algorithms
    data_arr = data.to_numpy()
    context = make_context().variables(data=data).build()

    # obtain the PAG that we should get using an oracle
    oracle = Oracle(causal_model.graph)
    fci_alg = FCI(ci_estimator=oracle)
    fci_alg.learn_graph(data, context)
    # true_pag = fci_alg.graph_

    ci_test = FisherZCITest()
    ci_test = GSquareCITest(data_type="discrete")
    clearn_ci_test = "gsq"

    # First run causallearn
    clearn_graph, edges = fci(data_arr, independence_test_method=clearn_ci_test, alpha=alpha)

    # Second run FCI
    fci_alg = FCI(ci_estimator=ci_test, alpha=alpha)
    fci_alg.learn_graph(data, context)
    dodiscover_graph = fci_alg.graph_

    # first compare the adjacency structure
    clearn_graph = clearn_to_graph(clearn_graph.graph, arr_idx=data.columns, graph_type="pag")
    cm = confusion_matrix_networks(dodiscover_graph, clearn_graph)

    dia = np.diag_indices(cm.shape[0])  # indices of diagonal elements
    dia_sum = sum(cm[dia])  # sum of diagonal elements
    off_dia_sum = np.sum(cm) - dia_sum  # subtract the diagonal sum from total array sum

    print(cm)
    assert off_dia_sum == 0

    # compare the SHD against the ground-truth graph
    assert (
        structure_hamming_dist(
            dodiscover_graph.sub_directed_graph(), clearn_graph.sub_directed_graph()
        )
        == 0
    )
    assert (
        structure_hamming_dist(
            dodiscover_graph.sub_bidirected_graph(), clearn_graph.sub_bidirected_graph()
        )
        == 0
    )
    assert (
        structure_hamming_dist(
            dodiscover_graph.sub_undirected_graph(), clearn_graph.sub_undirected_graph()
        )
        == 0
    )
    assert (
        structure_hamming_dist(dodiscover_graph.sub_circle_graph(), clearn_graph.sub_circle_graph())
        == 0
    )
