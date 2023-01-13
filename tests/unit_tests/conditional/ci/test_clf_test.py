import numpy as np
import pandas as pd
from flaky import flaky
from sklearn.ensemble import RandomForestClassifier

from dodiscover.ci import ClassifierCITest
from dodiscover.ci.simulate import nonlinear_additive_gaussian

seed = 12345
rng = np.random.RandomState(seed)


@flaky
def test_clf_with_gaussian_data():
    n_samples = 1000
    X = rng.randn(n_samples, 1)
    X1 = rng.randn(n_samples, 1)
    Y = X + X1 + 0.5 * rng.randn(n_samples, 1)
    Z = Y + 0.5 * rng.randn(n_samples, 1)

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])
    clf = RandomForestClassifier(random_state=rng)
    ci_estimator = ClassifierCITest(clf, random_state=rng)

    _, pvalue = ci_estimator.test(df, {"x"}, {"x1"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    assert pvalue > 0.05


@flaky
def test_clf_with_nonlinear_cos_additive():
    # need a decent number of samples to fit the classifiers
    n_samples = 4000

    # create input for the CI test
    X, Y, Z = nonlinear_additive_gaussian(model_type="ci", n_samples=n_samples, random_state=rng)
    df = pd.DataFrame(np.hstack((X, Y, Z)), columns=["x", "y", "z"])

    clf = RandomForestClassifier(random_state=rng)
    ci_estimator = ClassifierCITest(clf, random_state=rng)
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"}, {"z"})
    assert pvalue > 0.05

    # create input for the ind test
    X, Y, Z = nonlinear_additive_gaussian(model_type="ind", n_samples=n_samples, random_state=rng)
    df = pd.DataFrame(np.hstack((X, Y, Z)), columns=["x", "y", "z"])

    clf = RandomForestClassifier(random_state=rng, n_jobs=-1)
    ci_estimator = ClassifierCITest(clf, random_state=rng)
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"z"}, {"y"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    assert pvalue > 0.05

    # create input for the dep test
    X, Y, Z = nonlinear_additive_gaussian(model_type="dep", n_samples=n_samples, random_state=rng)
    df = pd.DataFrame(np.hstack((X, Y, Z)), columns=["x", "y", "z"])

    clf = RandomForestClassifier(random_state=rng, n_jobs=-1)
    ci_estimator = ClassifierCITest(clf, random_state=rng)
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, {"z"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    assert pvalue < 0.05


def test_clfcitest_with_bootstrap():
    """Test classifier CI test with bootstrap option."""
    # need a decent number of samples to fit the classifiers
    n_samples = 1000

    # create input for the CI test
    X, Y, Z = nonlinear_additive_gaussian(
        model_type="ci", n_samples=n_samples, random_state=rng, std=0.001
    )
    df = pd.DataFrame(np.hstack((X, Y, Z)), columns=["x", "y", "z"])

    clf = RandomForestClassifier(random_state=rng, n_jobs=-1)
    ci_estimator = ClassifierCITest(clf, random_state=rng, bootstrap=True, n_iter=2)
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"y"}, {"z"})
    assert pvalue > 0.05

    # create input for the dep test X -> Y <- Z
    X, Y, Z = nonlinear_additive_gaussian(
        model_type="dep", n_samples=n_samples, random_state=rng, std=0.001
    )
    clf = RandomForestClassifier(random_state=rng, n_jobs=-1)
    ci_estimator = ClassifierCITest(clf, random_state=rng, bootstrap=True, n_iter=2)
    df = pd.DataFrame(np.hstack((X, Y, Z)), columns=["x", "y", "z"])
    _, pvalue = ci_estimator.test(df, {"z"}, {"y"})
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, {"x"}, {"z"}, {"y"})
    assert pvalue < 0.05
