import numpy as np
import pytest

from causal_networkx import StructuralCausalModel
from causal_networkx.ci import (
    FisherZCITest,
    GSquareCITest,
    KernelCITest,
    Oracle,
    ParentChildOracle,
    PartialCorrelation,
)

seed = 12345
rng = np.random.RandomState(seed=seed)
func_uz = lambda: rng.negative_binomial(n=1, p=0.25)
func_uxy = lambda: rng.binomial(n=1, p=0.4)
func_x = lambda u_xy: 2 * u_xy
func_y = lambda x, u_xy, z: x + u_xy + z
func_z = lambda u_z: u_z

# construct the SCM and the corresponding causal graph
scm = StructuralCausalModel(
    exogenous={
        "u_xy": func_uxy,
        "u_z": func_uz,
    },
    endogenous={"x": func_x, "y": func_y, "z": func_z},
)

sample_df = scm.sample(n=100)
ground_truth_graph = scm.get_causal_graph()


@pytest.mark.parametrize(
    "ci_estimator",
    [
        KernelCITest(),
        GSquareCITest(),
        FisherZCITest(),
        PartialCorrelation(),
        Oracle(ground_truth_graph),
        ParentChildOracle(ground_truth_graph),
    ],
)
def test_ci_tests(ci_estimator):
    x = "x"
    y = "y"
    with pytest.raises(ValueError, match="The z conditioning set variables are not all"):
        ci_estimator.test(sample_df, x, y, z_covariates=["blah"])

    with pytest.raises(ValueError, match="The x and y variables are not both"):
        ci_estimator.test(sample_df, x, y_var="blah", z_covariates=["z"])
