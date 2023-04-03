import networkx as nx
import pytest

from dodiscover.ci import FisherZCITest, GSquareCITest, KernelCITest, Oracle
from dodiscover.constraint.utils import dummy_sample

ground_truth_graph = nx.DiGraph(
    [
        ("x", "y"),
        ("z", "y"),
    ]
)
sample_df = dummy_sample(ground_truth_graph)


@pytest.mark.parametrize(
    "ci_estimator",
    [
        KernelCITest(),
        GSquareCITest(),
        FisherZCITest(),
        Oracle(ground_truth_graph),
    ],
)
def test_ci_tests(ci_estimator):
    x = "x"
    y = "y"
    with pytest.raises(ValueError, match="The z conditioning set variables .* are not all"):
        ci_estimator.test(sample_df, {x}, {y}, z_covariates=["blah"])

    with pytest.raises(ValueError, match="The x variables.*are not all"):
        ci_estimator.test(sample_df, {"blah"}, y_vars={y}, z_covariates=["z"])

    with pytest.raises(ValueError, match="The y variables.*are not all"):
        ci_estimator.test(sample_df, {x}, y_vars={"blah"}, z_covariates=["z"])
