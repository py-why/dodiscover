"""
To run this, you'll need to have installed.

  * causal-learn

First, we fix some necessary simulation data. As of writing this benchmark
causal-learn does not support the input of an Oracle, so we will utilize
explicit CI tests. The goal of these two benchmarks is to evaluate the
PC algorithm and variants in dimensionality and sample size of the dataset
and compare the output in time/output relative to causal-learn.

Does two benchmarks

First, we increase the number of variables that the dataset contains
to look at the impact of dimensionality on the PC algorithm.

In the second benchmark, we increase the number of samples and evaluate
how the PC algorithm performs as a result of increasing sample size.
"""
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime

from dodiscover.constraint import PC
from causallearn.search.ConstraintBased.PC import pc_alg

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10**6  # number of microseconds in a second


def bench_scikit_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sklearn.tree import DecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeClassifier()
    clf.fit(X, Y).predict(X)
    delta = datetime.now() - tstart
    # stop time

    scikit_classifier_results.append(delta.seconds + delta.microseconds / mu_second)


def bench_scikit_tree_regressor(X, Y):
    """Benchmark with scikit-learn decision tree regressor"""

    from sklearn.tree import DecisionTreeRegressor

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = datetime.now() - tstart
    # stop time

    scikit_regressor_results.append(delta.seconds + delta.microseconds / mu_second)


if __name__ == "__main__":

    print("============================================")
    print("Warning: this is going to take a looong time")
    print("============================================")

    n = 10
    step = 10000
    n_samples = 10000
    dim = 10
    n_classes = 10
    for i in range(n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(n_samples)
        bench_scikit_tree_regressor(X, Y)

    xx = range(0, n * step, step)
    plt.figure("scikit-learn tree benchmark results")
    plt.subplot(211)
    plt.title("Learning with varying number of samples")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")

    scikit_classifier_results = []
    scikit_regressor_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 10

    dim = start_dim
    for i in range(0, n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        dim += step
        X = np.random.randn(100, dim)
        Y = np.random.randint(0, n_classes, (100,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(100)
        bench_scikit_tree_regressor(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title("Learning in high dimensional spaces")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of dimensions")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
