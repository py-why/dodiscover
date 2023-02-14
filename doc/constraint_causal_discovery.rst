.. _constraint_causal_discovery:

==================================
Constraint-based causal discovery
==================================

.. currentmodule:: dodiscover.constraint

The following are a set of methods intended for (non-parametric) structure learning
of causal graphs (i.e. causal discovery) given observational and/or interventional data
by checking constraints in the form of conditional independences (CI). At a high level
all constraint-based causal discovery, tests CI statements, where we use different CI
statistical tests to test the following null hypothesis:

:math:`H_0: X \perp Y | Z` and :math:`H_A: X \not\perp Y | Z`

For a given node ``X`` and ``Y`` in the underlying causal graph of interest, and
a conditioning set, ``Z``. CI tests can have a variety of different assumptions
that make one better than another in different settings and data assumptions.
For more information on the CI tests themselves, see :ref:`conditional_independence`.

Fundamental Assumptions of Constraint-Based Causal Discovery
------------------------------------------------------------
The fundamental assumptions of all algorithms in this section are the Markov property assumption
and the causal faithfulness assumption :footcite:`Spirtes1993` and :footcite:`Pearl_causality_2009`.
The Markov assumption states that all d-separation statements in the causal graph imply a
corresponding CI statement in the data. This is a core-assumption that users from graphical modeling
may be familiar with. On the other hand, the causal faithfulness assumption states that all
CI statements in the data map to a d-separation statement. That is, there are no accidental
CI that occur in the data, which are not represented by a d-separation statement in the underlying
causal graph. The causal faithfulness assumption is a very problematic assumption because in practice
one might have data that is very weakly dependent, such that a CI test under a specified :math:`\alpha`
level would fail to reject the null hypothesis and conclude the variables in question are CI. In higher
dimensions this can occur a large percentage of the time as demonstrated in :footcite:`uhler2013geometry`.

Tackling constraint-based causal discovery is a large and active area of research.

(Non-parametric) Markovian SCMs with Observational Data
-------------------------------------------------------
If one assumes that the underlying structural causal model (SCM) is Markovian,
then the Peter and Clarke (PC) algorithm has been shown to be sound and complete
for learning a completed partially directed acyclic graph (CPDAG) :footcite:`Meek1995`.

The PC algorithm and its variants assume Markovianity, which is also known as
causal-sufficiency in the literature. In other words, it assumes a lack of latent
confounders, where there is no latent variable that is a confounder of the observed data.

The PC algorithm learns a CPDAG in three stages:

1. skeleton discovery: This first phase is the process of leveraging CI tests to test
    edges for conditional independence.
2. unshielded triplet orientation:
3. deterministic path orientations:


(Non-parametric) Semi-Markovian SCMs with Observational Data
------------------------------------------------------------


(Non-parametric) SCMs with Interventional Data
----------------------------------------------


Choosing the conditioning sets
------------------------------
To describe.

Hyperparameters and controlling overfitting
-------------------------------------------
To describe.

Robust learning
---------------
Conservative orientations, etc.



