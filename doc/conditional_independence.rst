.. _conditional_independence:

========================
Conditional Independence
========================

.. currentmodule:: dodiscover.ci

Testing for invariances in the data, such as conditional independence (CI) can be represented graphically
in the form of d-separation statements under the causal faithfulness assumption. Given this, one
is interested in high-powered and well-controlled CI tests that can be used to test for CI in data.

CI tests are framed as a statistical hypothesis test, with the following null hypothesis for a given
pair of variables ``(X, Y)`` and a conditioning set ``Z`` (which may be empty):

:math:`H_0: X \perp Y | Z`, or written in terms of their distribution :math:`H_0: P(Y | X, Z) = P(Y | Z)`

Similarly, the alternative hypothesis is written as:

:math:`H_0: X \not\perp Y | Z`, or written in terms of their distribution :math:`H_0: P(Y | X, Z) \neq P(Y | Z)`

Then typically, one posits an acceptable Type I error rate (false positive), typically :math:`\alpha = 0.05`
and then either attempts to sample from the null distribution, or characterizes the asymptotic distribution
of the test statistic. In both approaches a pvalue is computed, which is compared to :math:`\alpha`. The pvalue
states the "probability that we observe our data (e.g. test statistic) under the null hypothesis". By rejecting
the null hypothesis, one claims that :math:`X \not\perp Y | Z`, so that X and Y are in fact (conditionally)
dependent given Z. Note that if one fails to reject the null hypothesis, it is simply by convention that we
claim X and Y are conditionally independent. It is not necessarily the case, and it is plausible that there
is a weak dependency that is unable to be captured by our proposed CI test, and/or data samples.

It is because of this reason, one would typically like the most powerful test given assumptions about
the data. With that in mind, there are various approaches to CI testing that are typically more powerful
with certain assumptions on the underlying data distribution.

Conditional Mutual Information
------------------------------
TBD.

Partial (Pearson) Correlation
-----------------------------
TBD.

Discrete, Categorical and Binary Data
-------------------------------------
TBD.

Kernel-Approaches
-----------------
TBD.

Classifier-based Approaches
---------------------------
TBD.