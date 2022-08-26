import dowhy.gcm as gcm
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from dodiscover.ci import FisherZCITest, GSquareCITest, KernelCITest, Oracle

seed = 12345
rng = np.random.RandomState(seed=seed)
func_z = rng.negative_binomial(n=1, p=0.25)
func_x = rng.binomial(n=1, p=0.4)
func_y = rng.binomial(n=1, p=0.2)
data = pd.DataFrame(data=dict(x=func_x, y=func_y, z=func_z))

# construct the SCM and the corresponding causal graph
graph = nx.DiGraph([("x", "y"), ("z", "y")])
causal_model = gcm.StructuralCausalModel(graph=graph)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)
sample_df = gcm.draw_samples(causal_model, num_samples=10)


@pytest.mark.parametrize(
    "ci_estimator",
    [
        KernelCITest(),
        GSquareCITest(),
        FisherZCITest(),
        Oracle(graph),
    ],
)
def test_ci_tests(ci_estimator):
    x = "x"
    y = "y"
    with pytest.raises(ValueError, match="The z conditioning set variables are not all"):
        ci_estimator.test(sample_df, {x}, {y}, z_covariates=["blah"])

    with pytest.raises(ValueError, match="The x variables are not all"):
        ci_estimator.test(sample_df, {"blah"}, y_vars={y}, z_covariates=["z"])

    with pytest.raises(ValueError, match="The y variables are not all"):
        ci_estimator.test(sample_df, {x}, y_vars={"blah"}, z_covariates=["z"])
