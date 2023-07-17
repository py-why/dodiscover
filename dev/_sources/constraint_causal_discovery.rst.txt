.. _constraint_causal_discovery:

==================================
Constraint-based causal discovery
==================================
.. automodule:: dodiscover.constraint
    :no-members:
    :no-inherited-members:
.. currentmodule:: dodiscover

Testing for invariances in the data, such as conditional independence (CI) can be represented graphically
in the form of d-separation statements under the causal faithfulness assumption. Given this, one
is interested in high-powered and well-controlled CI tests that can be used to test for CI in data.

The following are a set of methods intended for (non-parametric) structure learning
of causal graphs (i.e. causal discovery) given observational and/or interventional data
by checking constraints in the form of conditional independences (CI). At a high level, all constraint-based
causal discovery algorithms test CI statements, where we use different CI
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

The Markov assumption states that any d-separation statement implies a CI statement. This is a
core-assumption that users from graphical modeling may be familiar with. It is a
general assumption that connects graphical models to probabilistic models.

On the other hand, the causal faithfulness assumption states that all
CI statements in the data map to a d-separation statement. That is, there are no accidental
CI that occur in the data, which are not represented by a d-separation statement in the underlying
causal graph. The causal faithfulness assumption in theory is not an issue due to a theoretical result
showing that violations of faithfulness in causal diagrams have measure zero (i.e. they do not occur).

However, in practice the causal faithfulness assumption is a very problematic assumption because 
one might have data that is very weakly dependent, such that a CI test under a specified :math:`\alpha`
level would fail to reject the null hypothesis and conclude the variables in question are CI. violations
of "strong-faithfulness" occurs frequently and almost surely in higher
dimensions :footcite:`uhler2013geometry`.

Tackling violations of faithfulness in constraint-based causal discovery is a large and active
area of research.

(Non-parametric) Markovian SCMs with Observational Data
-------------------------------------------------------
If one assumes that the underlying structural causal model (SCM) is Markovian,
then the Peter and Clarke (PC) algorithm has been shown to be sound and complete
for learning a completed partially directed acyclic graph (CPDAG) :footcite:`Meek1995`.

The :class:`dodiscover.constraint.PC` algorithm and its variants assume Markovianity, which is
also known as causal-sufficiency in the literature. In other words, it assumes a lack of latent
confounders, where there is no latent variable that is a confounder of the observed data.

The PC algorithm learns a CPDAG in three stages:

1. skeleton discovery: This first phase is the process of leveraging CI tests to test
    edges for conditional independence. Along the way, connections of the graph are trimmed
    (when CI is detected) and the separating sets among pairs of variables are tracked.

    A separating set is a set of nodes in the graph that d-separate a pair of variables.
    Note that a pair of variables may contain many d-separators, and thus there may be
    many separating sets.
2. unshielded triplet orientation: This takes triplets on a path of the form ``X *-* Y *-* Z``,
    where the triplet path is "unshielded" meaning ``X`` and ``Z`` are not connected. Then
    it checks that ``Y`` is not in the separating set of X and Z. Given these two conditions,
    Y must be a collider and is oriented as ``X *-> Y <-* Z``. The stars in the path indicate
    that it can be any kind of edge endpoint (e.g. in a PAG it could be a circle endpoint edge).
3. deterministic path orientations: Once all colliders are oriented, there are a set of
    deterministic logical rules that allow us to orient more edges. In the PC algorithm,
    these are the so-called "Meek's orientation rules", which are 4 rules that are applied
    repeatedly until no more changes to the graph are made :footcite:`Meek1995`.

The resulting graph is an equivalence class of DAGs without latent confounders, the CPDAG.
For more information on CPDAGs, one can also see :class:`pywhy_graphs.CPDAG`.

(Non-parametric) Semi-Markovian SCMs with Observational Data
------------------------------------------------------------
If one assumes that the underlying SCM is Semi-Markovian, then the "Fast Causal Inference"
(FCI) algorithm has been shown to be sound and complete for learning a partial ancestral
graph (PAG) :footcite:`zhang2008ancestralgraphs,Zhang2008`.

The FCI algorithm and its variants assume Semi-Markovianity, which assumes the
possible presence of latent confounders and even selection bias in the observational data.

The :class:`dodiscover.constraint.FCI` algorithm follows the three stages of learning that the PC
algorithm does, but with a few minor modifications that we will outline here:

1. skeleton discovery: The skeleton discovery phase is now composed of two stages. The first
    stage is the same as the PC algorithm. The second phase, takes the output graph of the first
    phase and tries to orient colliders. This results in a PAG that can be queried for the
    potentially d-separating (PDS) sets for any pair of variables ``(X, Y)``. The skeleton
    discovery phase is restarted from scratch, but now the conditioning sets are chosen from
    the PDS sets. The PDS set approach is described in :footcite:`Colombo2012` and
    :footcite:`Spirtes1993`.
2. deterministic path orientations: The four orientation rules of the PC algorithm are still
    the same, but in the FCI case, we add an additional six orientation rules. The additional
    rules account for latent confounding and selection bias. Three of those rules
    only apply if we assume selection bias is present.

(Non-parametric) SCMs with Interventional Data
----------------------------------------------
When we have access to experimental data, there are multiple datasets corresponding to multiple
distributions (e.g. observational and different interventions), we can improve causal discovery.
If one assumes we have access to multiple distributions, one may know the targets of
each intervention, where one can apply the I-FCI algorithm to learn an Interventional-PAG
(I-PAG) :footcite:`Kocaoglu2019characterization`.

Alternatively, one may assume they do not know where the intervention was applied in each
distribution. In this case, one may apply the :math:`Psi`-FCI algorithm to learn a
:math:`Psi`-PAG :footcite:`Jaber2020causal`.

.. autosummary::
   :toctree: generated/

    constraint.PsiFCI

Choosing the conditioning sets
------------------------------
We briefly describe how ``dodiscover`` chooses conditioning sets, ``Z`` that are tested given
a pair of nodes ``(X, Y)``. The test we are doing is :math:`X \perp Y | Z`, where ``Z`` can
be the empty set. There are multiple strategies for choosing ``Z``.

.. autosummary::
   :toctree: generated/

   constraint.ConditioningSetSelection

Hyperparameters and controlling overfitting
-------------------------------------------
To describe.

Robust learning
---------------
Conservative orientations, etc.



