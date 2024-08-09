import bnlearn
import numpy as np
import pandas as pd
import networkx as nx
import pytest
from causallearn.search.ConstraintBased.PC import pc_alg
from pywhy_graphs.export import clearn_to_graph
from sklearn.preprocessing import LabelEncoder

from dodiscover import PC, make_context
from dodiscover.ci import GSquareCITest, KernelCITest
from dodiscover.metrics import structure_hamming_dist


# TODO: investigate why FisherZ test is different between causal-learn and dodiscover
# TODO: investigate why G^2 test for discrete data is different between causal-learn and
# XXX: dodiscover possibly from an error in our impleemntation?
@pytest.mark.parametrize(
    ["dataset", "ci_estimator", "clearn_test", "col_names", "categorical_cols"],
    [
        # [
        #     "titanic",
        #     FisherZCITest(),
        #     "fisherz",
        #     ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
        #     ["Sex", "Embarked"],
        # ],
        # [
        #     "sachs",
        #     FisherZCITest(),
        #     "fisherz",
        #     ["Erk", "Akt", "PKA", "Mek", "Jnk", "PKC", "Raf", "P38", "PIP3", "PIP2", "Plcg"],
        #     [],
        # ],
        # [
        #     "water",
        #     GSquareCITest("discrete"),
        #     "gsq",
        #     [
        #         "C_NI_12_00",
        #         "C_NI_12_15",
        #         "CKNI_12_00",
        #         "CKNI_12_15",
        #         "CBODD_12_15",
        #         "CBODD_12_00",
        #         "CNOD_12_00",
        #         "CBODN_12_00",
        #         "CKND_12_15",
        #         "CKND_12_00",
        #         "CKNN_12_00",
        #         "CNOD_12_15",
        #         "CNON_12_00",
        #         "CBODN_12_15",
        #         "CKNN_12_15",
        #         "CNON_12_15",
        #         "C_NI_12_30",
        #         "CKNI_12_30",
        #         "CBODD_12_30",
        #         "CKND_12_30",
        #         "CNOD_12_30",
        #         "CBODN_12_30",
        #         "CKNN_12_30",
        #         "CNON_12_30",
        #         "C_NI_12_45",
        #         "CKNI_12_45",
        #         "CBODD_12_45",
        #         "CKND_12_45",
        #         "CNOD_12_45",
        #         "CBODN_12_45",
        #         "CKNN_12_45",
        #         "CNON_12_45",
        #     ],
        #     [],
        # ],
        [
            "sprinkler",
            GSquareCITest("binary"),
            "gsq",
            ["Cloudy", "Sprinkler", "Rain", "Wet_Grass"],
            [],
        ],
        [
            "asia",
            GSquareCITest("binary"),
            "gsq",
            ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"],
            [],
        ],
        [
            "stormofswords",
            KernelCITest(),
            "kci",
            ["source", "target", "weight"],
            ["source", "target"],
        ],  # discrete and categorical
        ["random", GSquareCITest("binary"), "gsq", ["A", "B", "C", "D", "E"], []],
    ],
)
def test_pc_against_causallearn(dataset, ci_estimator, clearn_test, col_names, categorical_cols):
    if dataset == "random":
        df = pd.DataFrame(np.random.randint(0, 2, (1000, len(col_names))), columns=col_names)
    else:
        df = bnlearn.import_example(dataset, n=1000, verbose=False)

    # only keep non-nan rows
    print(df.columns)
    df = df[col_names]
    df.dropna(inplace=True)
    print(df.shape)
    print(df.head(3))

    # run encoder
    enc = LabelEncoder()
    for col in categorical_cols:
        df[col] = enc.fit_transform(df[[col]].to_numpy().ravel())

    # make context and setup for dodiscover
    alpha = 0.05
    context = make_context().variables(data=df).build()

    pcalg = PC(ci_estimator=ci_estimator, alpha=alpha)
    pcalg.learn_graph(df, context)
    pywhy_graph = pcalg.graph_

    # now run causal-learn
    data = df.to_numpy()
    nodes = df.columns.tolist()
    clearn_graph = pc_alg(
        data,
        node_names=nodes,
        alpha=alpha,
        indep_test=clearn_test,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
    )

    # convert to pywhy graph
    nodes = [node.get_name() for node in clearn_graph.G.nodes]
    clearn_pywhy_graph = clearn_to_graph(clearn_graph.G.graph, arr_idx=nodes, graph_type="cpdag")
    print(pywhy_graph)
    print(clearn_pywhy_graph)
    print(clearn_graph.G.graph)

    print(pywhy_graph.sub_directed_graph())
    print(clearn_pywhy_graph.sub_directed_graph())
    print(pywhy_graph.sub_undirected_graph())
    print(clearn_pywhy_graph.sub_undirected_graph())

    shd = structure_hamming_dist(
        clearn_pywhy_graph.sub_directed_graph(), pywhy_graph.sub_directed_graph()
    )

    # now we will compare the two graphs
    assert nx.is_isomorphic(
        clearn_pywhy_graph.sub_directed_graph(), pywhy_graph.sub_directed_graph()
    )
    assert shd == 0
    assert nx.is_isomorphic(
        clearn_pywhy_graph.sub_undirected_graph(), pywhy_graph.sub_undirected_graph()
    )
