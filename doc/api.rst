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

Graph protocols
===============

.. currentmodule:: dodiscover
.. autosummary::
   :toctree: generated/

   Graph
   EquivalenceClass


Context for causal discovery
============================

Rather than just data, in many cases structure learning
has additional "context", in the form of apriori knowledge of
the structure, or additional datasets from different environments.
All structure learning algorithms in ``dodiscover`` pass in a ``Context``
object rather than just data. See docs for ``Context`` for more information.

.. currentmodule:: dodiscover
.. autosummary::
   :toctree: generated/

   Context

Constraint-based structure learning
===================================

.. currentmodule:: dodiscover.constraint
.. autosummary::
   :toctree: generated/

   LearnSkeleton
   LearnSemiMarkovianSkeleton
   SkeletonMethods
   PC

Comparing causal discovery algorithms
=====================================

.. currentmodule:: dodiscover.metrics
.. autosummary::
   :toctree: generated/

   confusion_matrix_networks
   structure_hamming_dist
