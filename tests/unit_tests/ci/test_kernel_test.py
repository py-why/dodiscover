import numpy as np
import pandas as pd
import pytest

from dodiscover.ci import KernelCITest

seed = 12345
ci_params = {
    "rbf_approx": KernelCITest(),
    "rbf": KernelCITest(approx_with_gamma=False),
    "linear_approx": KernelCITest(
        kernel_x="linear",
        kernel_y="linear",
        kernel_z="linear",
    ),
    "linear": KernelCITest(
        kernel_x="linear", kernel_y="linear", kernel_z="linear", approx_with_gamma=False
    ),
    "polynomial_approx": KernelCITest(
        kernel_x="polynomial",
        kernel_y="polynomial",
        kernel_z="polynomial",
    ),
    "polynomial": KernelCITest(
        kernel_x="polynomial", kernel_y="polynomial", kernel_z="polynomial", approx_with_gamma=False
    ),
}


@pytest.mark.parametrize("ci_estimator", ci_params.values(), ids=ci_params.keys())
def test_kci_with_gaussian_data(ci_estimator):
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    X1 = rng.randn(300, 1)
    Y = X + X1 + 0.5 * rng.randn(300, 1)
    Z = Y + 0.5 * rng.randn(300, 1)

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    assert pvalue > 0.05


def test_kci_errors():
    with pytest.raises(ValueError, match="The kernels that are currently supported"):
        KernelCITest(kernel_x="gauss")
