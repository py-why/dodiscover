import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dodiscover.ci import ClassifierCMITest
from dodiscover.ci.ccmi_test import f_divergence_score, kl_divergence_score

seed = 12345


def test_ccmi_with_nonlinear_data_null():
    """Test kl-divergence with computing null distribution."""

    threshold = None
    alpha = 0.15
    rng = np.random.default_rng(seed)

    clf = RandomForestClassifier(random_state=seed)
    ci_estimator = ClassifierCMITest(
        clf=clf,
        metric=kl_divergence_score,
        n_shuffle_nbrs=3,
        n_shuffle=10,
        threshold=threshold,
        test_size=0.1,
        n_jobs=-1,
        random_seed=seed,
    )

    # X -> Y <- X1; Y -> Z
    n_samples = 1000
    X = rng.standard_normal((n_samples, 1))
    X1 = rng.uniform(low=0.0, high=1.0, size=(n_samples, 1))
    Y = X + X1 + 1e-5 * rng.standard_normal((n_samples, 1))
    Z = Y + 1e-5 * rng.standard_normal((n_samples, 1))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    print(pvalue)
    assert pvalue > alpha
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    print(pvalue)
    assert pvalue > alpha
    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, {"z"})
    print(pvalue)
    assert pvalue < alpha
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    print(pvalue)
    assert pvalue < alpha


def test_ccmi_with_nonlinear_data_thresh():
    """Test thresholding with f-divergence score."""
    threshold = 0.01
    rng = np.random.default_rng(seed)

    clf = RandomForestClassifier(random_state=seed)
    ci_estimator = ClassifierCMITest(
        clf=clf,
        metric=f_divergence_score,
        n_shuffle_nbrs=5,
        n_shuffle=10,
        threshold=threshold,
        test_size=0.1,
        n_jobs=-1,
        random_seed=seed,
    )

    # X -> Y <- X1; Y -> Z
    n_samples = 1000
    X = rng.standard_normal((n_samples, 1))
    X1 = rng.uniform(low=0.0, high=1.0, size=(n_samples, 1))
    Y = X + X1 + 1e-5 * rng.standard_normal((n_samples, 1))
    Z = Y + 1e-5 * rng.standard_normal((n_samples, 1))

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    print(pvalue)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"}, {"z"})
    print(pvalue)
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    print(pvalue)
    assert pvalue < 0.05
