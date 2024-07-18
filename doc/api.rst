.. _api_ref:

###
API
###

:py:mod:`dodiscover`:

.. automodule:: dodiscover
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of dodiscover, grouped thematically by analysis
stage.


Constraint-based structure learning
===================================

.. currentmodule:: dodiscover.constraint
.. autosummary::
   :toctree: generated/

   LearnSkeleton
   LearnSemiMarkovianSkeleton
   LearnInterventionSkeleton
   ConditioningSetSelection
   PC
   FCI
   PsiFCI


Order-based structure learning
===================================

.. currentmodule:: dodiscover.toporder
.. autosummary::
   :toctree: generated/

   CAM
   SCORE
   DAS
   NoGAM

Comparing causal discovery algorithms
=====================================

.. currentmodule:: dodiscover.metrics
.. autosummary::
   :toctree: generated/

   confusion_matrix_networks
   structure_hamming_dist
   toporder_divergence


Typing
======

We define some custom types to allow 3rd party packages
to work with ``mypy``.

.. currentmodule:: dodiscover.typing
.. autosummary::
   :toctree: generated/

   Column
   SeparatingSet
   NetworkxGraph


Graph protocols
===============

.. currentmodule:: dodiscover
.. autosummary::
   :toctree: generated/

   Graph
   EquivalenceClass

********************
 Conditional Testing
********************
Dodiscover experimentally provides an interface for conditional independence
testing and conditional discrepancy testing (also known as k-sample conditional
independence testing).

Conditional Independence Testing
================================

Testing for conditional independence among variables is a core part
of many causal inference procedures, such as constraint-based structure
learning.

.. currentmodule:: dodiscover.ci
.. autosummary::
   :toctree: generated/

   BaseConditionalIndependenceTest
   Oracle
   KernelCITest
   GSquareCITest
   FisherZCITest
   ClassifierCITest
   CMITest
   ClassifierCMITest
   CategoricalCITest

Conditional k-sample testing
============================

Testing for conditional discrepancy among variables is a core part
of many causal inference procedures, such as constraint-based structure
learning.

.. currentmodule:: dodiscover.cd
.. autosummary::
   :toctree: generated/

   BaseConditionalDiscrepancyTest
   KernelCDTest
   BregmanCDTest

Utilities
=========

Testing for conditional discrepancy among variables is a core part
of many causal inference procedures, such as constraint-based structure
learning.

.. currentmodule:: dodiscover.ci.kernel_utils
.. autosummary::
   :toctree: generated/

   compute_kernel
   corrent_matrix
   von_neumann_divergence
   f_divergence_score
   kl_divergence_score


Other utilities include support for Monte-Carlo testing for conditional
independence.

.. currentmodule:: dodiscover.ci.monte_carlo
.. autosummary::
   :toctree: generated/

   generate_knn_in_subspace
   restricted_nbr_permutation

**The following API is for internal development and is completely experimental.**

Context for causal discovery
============================

Rather than just data, in many cases structure learning
has additional "context", in the form of apriori knowledge of
the structure, or additional datasets from different environments.
All structure learning algorithms in ``dodiscover`` pass in a ``Context``
object rather than just data. One should use our builder ``make_context``
API for construction of the Context class. 

See docs for ``Context`` and ``make_context`` for more information.

.. currentmodule:: dodiscover
.. autosummary::
   :toctree: generated/

   make_context
   ContextBuilder
   InterventionalContextBuilder
   context.Context
