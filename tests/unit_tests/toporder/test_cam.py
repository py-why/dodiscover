import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dodiscover import make_context
from dodiscover.metrics import structure_hamming_dist, toporder_divergence
from dodiscover.toporder.cam import CAM
from dodiscover.toporder.utils import full_DAG, orders_consistency


# -------------------- Fixtures -------------------- #
@pytest.fixture
def dummy_sample():
    """
    Ground truth:  [[0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 0, 0]]
    """
    X = np.array(
        [
            [-0.839573121, -0.277701505, -0.430597219, -0.060861940],
            [0.310605822, -2.599033223, -1.014627287, 0.378551638],
            [-0.370546332, -1.203729713, -1.825172114, 0.858230621],
            [-2.699795437, 0.008554048, 0.981706806, -0.004763567],
            [-0.645773678, -1.244333225, 2.139405701, -0.685640790],
            [-0.042450689, -1.004666403, 0.305874333, 0.286281663],
            [-0.470815585, -0.424620043, -0.088586494, -0.728520832],
            [-0.127119621, -1.365008612, 0.572606554, -0.395512040],
            [-0.923604918, 1.189839981, -1.636338739, 1.261349131],
            [-1.122439494, 0.543208240, -0.117069594, -0.415525935],
            [-0.963623192, 1.515937821, -0.765983319, 0.162209738],
            [-0.938541331, -0.228746624, 0.100970040, -0.693817970],
            [-1.333079291, 0.219403190, -0.259291759, -0.268546737],
            [-1.302912366, -0.533871530, -1.523940135, 1.012304483],
            [1.073299231, -1.788119910, 0.076554379, -0.193270481],
            [-2.642256864, -3.380057237, 1.196454910, 0.055111126],
            [-2.408092569, 0.203764866, 0.604369557, -0.674611861],
            [-0.098306882, -1.139202301, -1.154991864, 0.110190182],
            [-1.037167528, 1.969153124, -2.195065802, 0.896123077],
            [1.523183183, -1.726094165, -0.383925779, -0.089046831],
            [-0.636641129, 2.172974227, 0.224130410, -0.421800221],
            [-2.571874050, -0.101080426, 1.831829673, -0.285606201],
            [-1.750336087, 0.275304662, -1.366555274, 0.759718588],
            [-1.359723177, -1.181449274, 0.847686742, -0.270815178],
            [-1.092667077, -1.332806070, 1.116977798, -0.403352904],
            [0.308975322, -0.892726394, -0.167811519, -0.359333250],
            [-0.737393775, 0.790620176, 0.075744844, -0.647631974],
            [-2.176587940, 1.741070728, 0.714511667, -0.255168693],
            [0.877342839, -2.060444475, -1.004183715, 0.156282481],
            [-0.705755883, 1.477512468, -0.048488009, -0.395428949],
            [-3.356961954, -0.001123703, 1.434055406, -0.158521890],
            [-0.967569746, -1.448572545, 1.491023319, -0.072412350],
            [-1.589294016, -0.608877651, 0.597702888, -0.634250794],
            [-3.015157895, -0.295313339, 0.993359667, -0.324542910],
            [-1.122132650, -0.207544016, -0.193527564, -0.071519482],
            [-0.583705243, 1.307028345, -0.624839165, 0.294043038],
            [-1.883144611, -0.877693431, 2.058046514, -0.564343765],
            [-1.123296529, -0.473336439, -0.646480082, 0.202386395],
            [-2.367473742, 1.035084041, 1.533223227, -0.293675755],
            [-2.226456072, 1.320761589, 2.360583505, -0.774136536],
            [0.566644360, -1.362261354, 0.115061341, -0.281491750],
            [-0.335850791, 1.802044114, -1.237563762, 0.781928189],
            [-1.145455310, -0.403688072, -1.199567773, 0.596284347],
            [-2.672082227, -0.311260803, 0.905745894, -0.462103798],
            [-2.437789754, 0.147107007, 0.793553023, -0.629310082],
            [-2.373920781, 1.815620721, 1.679093156, 0.014996959],
            [-1.046924954, -1.424198472, 1.520208099, -0.032027423],
            [1.093059667, -2.332160912, -0.626965236, -0.131026893],
            [-0.225172953, -1.404801798, 0.524995366, -0.301736297],
            [-1.958224930, 1.694781609, 1.042384422, -0.087328916],
            [-0.024960619, 2.431748766, -0.981844758, 0.391344599],
            [0.252626346, -1.084801502, -0.149212095, -0.190797186],
            [0.606688957, -1.186083345, -0.650859427, -0.039576786],
            [-1.380819167, 1.403118305, -2.018894040, 1.016465100],
            [-2.581574031, 0.818861200, 0.729701937, -0.197394538],
            [-1.346638512, 0.100801746, -0.755724873, 0.117037842],
            [0.777483405, 0.770711406, -3.350778096, 0.747827309],
            [-1.555713074, 0.266937342, -1.223729700, 0.916256044],
            [-1.942841173, 1.285054506, 0.670483440, -0.637756333],
            [-1.900153344, 1.319635886, 0.564863699, -0.144524778],
            [-1.698271684, -0.340446115, 0.590809155, -0.505020973],
            [-1.186021352, -1.451510752, 1.547947860, 0.054982207],
            [-1.651768745, -1.067190567, 1.114559651, -0.494704236],
            [-2.124261987, -0.086312211, -1.582070647, 0.682998499],
            [-2.224634830, -0.699313432, 0.833965103, 0.009331806],
            [-2.359252335, 1.499530460, 1.519591298, -0.421090776],
            [-1.330045121, -0.359059430, -1.845437526, 1.174578169],
            [-0.602016153, -0.152355877, 3.692981177, -1.450500093],
            [-0.445155598, -0.437528377, 0.107226389, -0.439652967],
            [-2.153078451, 0.377040793, -1.930063763, 0.917191774],
            [-1.552542663, -0.223127490, 3.391570624, -1.266882952],
            [-1.475411548, 0.624069020, -2.044359173, 1.322692925],
            [-0.953616350, -0.937442786, 0.480845758, -0.609066310],
            [-1.484881907, 0.765424791, -0.978146493, 0.150888727],
            [-1.238649496, 0.512099423, -0.849546348, 0.059745938],
            [1.076467888, -2.041548555, 0.146570213, -0.283719583],
            [-0.659625642, -0.059244488, -0.540244144, -0.181741932],
            [-2.620588859, 1.789910920, 1.411626682, -0.655568280],
            [-1.957950942, -0.531252841, 0.825882661, 0.284562924],
            [-2.730679543, 0.176578095, 2.228774978, -0.266674091],
            [-1.858454741, 0.776888126, -1.960306318, 1.007143344],
            [-2.443972960, 0.280944745, 2.492478774, -1.088537047],
            [-1.389120868, 1.227356071, 0.583647368, 0.080900717],
            [-1.149132172, 0.834687238, -0.813570101, 0.022291579],
            [-1.374562294, 0.978715123, -1.704717257, 0.954258580],
            [-2.013843044, -0.926839642, 1.364829455, -0.153602066],
            [-1.774956858, 0.373195342, -1.486428689, 0.699396514],
            [-0.184860788, -2.972180860, 0.260938612, -0.235503163],
            [-1.251798479, 1.695334713, 0.321958695, -0.690529829],
            [1.146869763, -1.518644272, -0.523074032, 0.047112852],
            [-0.628250791, -0.100670367, -0.434009608, -0.350679160],
            [-1.232690859, 2.586234508, 0.647387022, -0.133229058],
            [-0.713577309, -1.476375817, 0.777031425, -0.147356364],
            [-0.804966718, 2.194410748, -1.641547938, 1.208676146],
            [0.262763163, -1.458057378, -0.713583677, -0.201490961],
            [-1.072651091, 0.982488481, -1.702774708, 0.957539390],
            [-2.248814981, 0.548801852, 0.552970447, -0.405830529],
            [-1.883670378, -0.819602933, 1.367503036, -0.424572056],
            [-2.059156478, -0.183935035, -1.373185103, 0.489411760],
            [0.052749913, -0.962538270, 0.196976968, -0.608109659],
        ]
    )
    return pd.DataFrame(X)


@pytest.fixture
def dummy_groundtruth():
    """
    Ground truth associated to dummy_sample dataset
    """
    A = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0]])
    return A


@pytest.fixture
def dummy_dense():
    """
    Dense adjacency matrix associated to order = [2, 1, 3, 0]
    """
    A = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0]])
    return A


# -------------------- Unit Tests -------------------- #
def test_given_dataset_when_fitting_CAM_then_shd_larger_equal_dtop(dummy_sample, dummy_groundtruth):
    model = CAM()
    context = make_context().variables(observed=dummy_sample.columns).build()
    model.fit(dummy_sample, context)
    A_pred = nx.to_numpy_array(model.graph_)
    order_pred = model.order_

    true_graph = nx.from_numpy_array(dummy_groundtruth, create_using=nx.DiGraph)
    shd = structure_hamming_dist(
        true_graph=true_graph,
        pred_graph=nx.from_numpy_array(A_pred, create_using=nx.DiGraph),
        double_for_anticausal=False,
    )
    d_top = toporder_divergence(true_graph, order_pred)
    assert shd >= d_top


def test_given_dag_and_dag_without_leaf_when_fitting_then_order_estimate_is_consistent(
    dummy_sample,
):
    order_gt = [2, 1, 3, 0]
    model = CAM()
    context = make_context().variables(observed=dummy_sample.columns).build()
    model.fit(dummy_sample, context)
    order_full = model.order_

    pruned_dummy_sample = dummy_sample[order_gt[:-1]]
    pruned_context = make_context().variables(observed=pruned_dummy_sample.columns).build()
    model.fit(pruned_dummy_sample, pruned_context)
    order_noleaf = model.order_
    assert orders_consistency(order_full, order_noleaf)


def test_given_dataset_and_rescaled_dataset_when_fitting_then_returns_equal_output(dummy_sample):
    # Rescale the input and test consistency of the output
    model = CAM()
    context = make_context().variables(observed=dummy_sample.columns).build()
    model.fit(dummy_sample, context)
    A = nx.to_numpy_array(model.graph_)
    model.fit(dummy_sample * 2, context)
    A_rescaled = nx.to_numpy_array(model.graph_)
    assert np.allclose(A, A_rescaled)


def test_given_dataset_and_dataset_with_permuted_column_when_fitting_then_return_consistent_outputs(
    dummy_sample,
):
    model = CAM()
    context = make_context().variables(observed=dummy_sample.columns).build()

    # permute sample columns
    permutation = [1, 3, 0, 2]
    permuted_sample = dummy_sample[permutation]  # permute pd.DataFrame columns

    # Run inference on original and permuted data
    model.fit(permuted_sample, context)
    A_permuted = nx.to_numpy_array(model.graph_)
    order_permuted = model.order_
    model.fit(dummy_sample, context)
    A = nx.to_numpy_array(model.graph_)
    order = model.order_

    # Match variables order
    back_permutation = [2, 0, 3, 1]
    A_permuted = A_permuted[:, back_permutation]
    A_permuted = A_permuted[back_permutation, :]

    # permutation_order with correct variables name
    permutation_dict = {k: p for k, p in enumerate(permutation)}
    order_permuted = [permutation_dict[o] for o in order_permuted]
    assert order_permuted == order
    assert np.allclose(A_permuted, A)


def test_given_adjacency_when_pruning_then_returns_dag_with_context_included_edges(dummy_sample):
    model = CAM()
    context = make_context().variables(observed=dummy_sample.columns).build()
    model.fit(dummy_sample, context)
    A = nx.to_numpy_array(model.graph_)
    order = model.order_
    A_dense = full_DAG(order)
    d = len(dummy_sample.columns)
    edges = []  # include all edges in A_dense and not in A
    for i in range(d):
        for j in range(d):
            if A_dense[i, j] == 1 and A[i, j] == 0:
                edges.append((i, j))
    included_edges = nx.empty_graph(len(dummy_sample.columns), create_using=nx.DiGraph)
    included_edges.add_edges_from(edges)
    context = make_context(context).edges(include=included_edges).build()
    model.fit(dummy_sample, context)
    A_included = nx.to_numpy_array(model.graph_)
    assert np.allclose(A_dense, A_included)


def test_given_adjacency_when_pruning_with_pns_then_returns_dag_with_context_included_edges(
    dummy_sample,
):
    model = CAM(pns=True)
    context = make_context().variables(observed=dummy_sample.columns).build()
    model.fit(dummy_sample, context)
    A = nx.to_numpy_array(model.graph_)
    order = model.order_
    A_dense = full_DAG(order)
    d = len(dummy_sample.columns)
    edges = []  # include all edges in A_dense and not in A
    for i in range(d):
        for j in range(d):
            if A_dense[i, j] == 1 and A[i, j] == 0:
                edges.append((i, j))
    included_edges = nx.empty_graph(len(dummy_sample.columns), create_using=nx.DiGraph)
    included_edges.add_edges_from(edges)
    context = make_context(context).edges(include=included_edges).build()
    model.fit(dummy_sample, context)
    A_included = nx.to_numpy_array(model.graph_)
    assert np.allclose(A_dense, A_included)
