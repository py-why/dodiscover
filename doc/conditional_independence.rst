.. _conditional_independence:

============
Independence
============

.. currentmodule:: dodiscover.ci

Probabilistic independence among two random variables is when the realization of one
variable does not affect the distribution of the other variable. It is a fundamental notion
in probability and statistics that is used to determine if information about some variables
can be gleaned from observations of other variables. Independence can be tested statistically
in the form of unconditional (or marginal) independence, or conditional independence (CI).
In the following, we present a brief overview of the various approaches to CI testing, where
the marginal case is a special case when the conditioning set is empty.

Conditional independence (CI) tests are framed as a statistical hypothesis test, with the following null hypothesis for a given
pair of variables ``(X, Y)`` and a conditioning set ``Z`` (which may be empty). The null hypothesis is that X and Y are statistically
independent given a conditioning set Z, i.e.,

:math:`H_0: X \perp Y | Z`, or written in terms of their distribution :math:`H_0: P(Y | X, Z) = P(Y | Z)`

Similarly, the alternative hypothesis is written as:

:math:`H_A: X \not\perp Y | Z`, or written in terms of their distribution :math:`H_A: P(Y | X, Z) \neq P(Y | Z)`

Then typically, one posits an acceptable upper bound on the Type I error rate (false positive), typically :math:`\alpha = 0.05`
and then either attempts to sample from the null distribution, or characterizes the asymptotic distribution
of the test statistic. In both approaches a pvalue is computed, which is compared to :math:`\alpha`. The pvalue
states the "probability of observing a test-statistic at least as extreme as our observed test-statistic in null distribution". By rejecting
the null hypothesis, one claims that :math:`X \not\perp Y | Z`, so that X and Y are in fact (conditionally)
dependent given Z.

Note that if one fails to reject the null hypothesis, we cannot accept the alternative hypothesis of
independence strictly speaking. However, in practice in many settings, such as in causal discovery,
we would still conclude that they are independent. It is not necessarily the case though, and
it is plausible that there is a weak dependency that is unable to be captured by our proposed CI test, and/or data samples.

It is because of this reason, one would typically like the most powerful test given assumptions about
the data. With that in mind, there are various approaches to CI testing that are typically more powerful
with certain assumptions on the underlying data distribution.

Conditional Mutual Information
------------------------------
Conditional mutual information (CMI) is a general formulation of CI, where CMI is defined as
:math::
    
    \\int log \frac{p(x, y | z)}{p(x | z) p(y | z)}

As we can see, CMI is equal to 0, if and only if :math:`p(x, y | z) = p(x | z) p(y | z)`, which
is exactly the definition of CI. CMI is completely non-parametric and thus requires no assumptions
on the underlying distributions. Unfortunately, CMI is notoriously difficult to estimate. There are
various proposals in the literature for estimating CMI, which we summarize here:

- The Kraskov, Stogbauer and Grassberger (KSG) estimate approach estimates mutual information
  via nearest-neighbor statistics :footcite:`kraskov_estimating_2004`. It computes nearest-neighbors
  using a kNN algorithm. It was generalized to CMI by :footcite:`frenzel_partial_2007`.
  This class of estimators is asymptotically correct, meaning if we had an infinite amount of data
  we would obtain the true value of the CMI. However, it relies on statistics generated from the k-NN,
  which if we implement the naive approach using a KDTree, then it generally suffers in high-dimensions.
  In our examples, we see it suffer with dimensionality > 4 or 5.

  Estimates of CMI can be converted into a CI hypothesis test by permutation testing :footcite:`Runge2018cmi`.
  One can generate estimated samples from the null distribution by permuting samples in an intelligent manner
  and then the CMI value generated from the actual observed data can be compared to the CMI values computed
  on the permutated datasets to estimate a pvalue.

  It is worth noting that if one has good estimates of nearest-neighbors using for example a model that
  is effective in high-dimensions, then the KSG estimator for CMI may still be effective. For example,
  one can use variants of Random Forests to generate adaptive nearest-neighbor estimates in high-dimensions
  or on manifolds, such that the KSG estimator is still powerful.

.. autosummary::
   :toctree: generated/

    CMITest

- The Classifier Divergence approach estimates CMI using a classification model.

.. autosummary::
   :toctree: generated/

    ClassifierCMITest

- Direct posterior estimates can be implemented with a classification model by directly
  estimating :math:`P(y|x)` and :math:`P(y|x,z)`, which can be used as plug-in estimates
  to the equation for CMI.

Partial (Pearson) Correlation
-----------------------------
Partial correlation based on the Pearson correlation is equivalent to CMI in the setting
of normally distributed data. Computing partial correlation is fast and efficient and
thus attractive to use. However, this **relies on the assumption that the variables are Gaussiany**,
which may be unrealistic in certain datasets.

.. autosummary::
   :toctree: generated/

    FisherZCITest

Discrete, Categorical and Binary Data
-------------------------------------
If one has discrete data, then the test to use is based on Chi-square tests. The :math:`G^2`
class of tests will construct a contingency table based on the number of levels across
each discrete variable. An exponential amount of data is needed for increasing levels
for a discrete variable.

.. autosummary::
   :toctree: generated/

    GSquareCITest

Kernel-Approaches
-----------------
Kernel-based tests are attractive since they are semi-parametric and use kernel-based ideas
that have been shown to be robust in the machine-learning field. The Kernel CI test is a test
that computes a test statistic from kernels of the data and uses permutation testing to
generate samples from the null distribution :footcite:`Zhang2011`, which are then used to
estimate a pvalue.

.. autosummary::
   :toctree: generated/

    KernelCITest

Classifier-based Approaches
---------------------------
Another suite of approaches that rely on permutation testing is the classifier-based approach.

By shuffling the data, one can setup a hypothesis test for CI based on the
predicted probabilities from a classification-model. Intuitively, if the shuffled data is similar 
to the unshuffled data, such that the classification-model achieves non-trivial performance
(e.g. >50\% accuracy on a balanced dataset), then one fails to reject the null hypothesis and would
state that the original data was in fact CI :footcite:`Sen2017model`.

When performing marginal independence testing between two sets of variables, ``X`` and ``Y``,
it is sufficient to shuffle data by just permuting either ``X`` rows or ``Y`` rows uniformly.
When performing CI testing conditioned on a third set of variables ``Z``, one must perform what is known
as conditional shuffling. One computes the nearest-neighbors in the ``Z`` subspace and then
permutes rows based on samples that are close in ``Z`` subspace, which
helps maintain dependence between (X, Z) and (Y, Z) (if it exists), but generates a
conditionally independent dataset.


.. autosummary::
   :toctree: generated/

    ClassifierCITest

=======================
Conditional Discrepancy
=======================

.. currentmodule:: dodiscover.cd

Conditional discrepancy (CD) is another form of conditional invariances that may be exhibited by data. The
general question is whether or not the following two distributions are equal:

:math:`P_{i=j}(y|x) =? P_{i=k}(y|x)`

where :math:`P_i(.)` denote the distribution that explicitly comes from
a different group, or environment, denoted by the discrete indices :math:`i`. This is also
known in some cases as conditional k-sample testing, if there are a finite k number of groups
for :math:`P_i`. CD testing is important because it detects other kinds of invariances besides
CI. 

Discrete, Categorical and Binary Data
-------------------------------------
If one has entirely discrete data, then the problem of CD can be converted into a CI test that
leverages the Chi-square class of tests. Since ``y`` and ``x`` are discrete and so are the
indices of the distribution, one can convert the CD test:

:math:`P_{i=j}(y|x) =? P_{i=k}(y|x)` into the CI test :math:`P(y|x,i) = P(y|x)`, which can
be tested with the Chi-square CI tests.

Kernel-Approaches
-----------------
Kernel-based tests are attractive since they are semi-parametric and use kernel-based ideas
that have been shown to be robust in the machine-learning field. The Kernel CD test is a test
that computes a test statistic from kernels of the data and uses a weighted permutation testing
based on the estimated propensity scores to generate samples from the null distribution
:footcite:`Park2021conditional`, which are then used to estimate a pvalue.

.. autosummary::
   :toctree: generated/

    KernelCDTest

Bregman-Divergences
-------------------
The Bregman CD test is a divergence-based test
that computes a test statistic from estimated Von-Neumann divergences of the data and uses a
weighted permutation testing based on the estimated propensity scores to generate samples from the null distribution
:footcite:`Yu2020Bregman`, which are then used to estimate a pvalue.

.. autosummary::
   :toctree: generated/

    BregmanCDTest
