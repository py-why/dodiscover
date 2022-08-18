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

Constraint-Based Structure Learning
===================================

.. currentmodule:: dodiscover.ci
.. autosummary::
   :toctree: generated/

   PC

Conditional Independence Testing
================================

Testing for conditional independence among variables is a core part
of many causal inference procedures, such as constraint-based structure
learning.

.. currentmodule:: dodiscover.ci
.. autosummary::
   :toctree: generated/

   Oracle
   BaseConditionalIndependenceTest
   KernelCITest
   GSquareCITest
   FisherZCITest

Graph Protocols
===============

.. currentmodule:: dodiscover
.. autosummary::
   :toctree: generated/

   GraphProtocol
   EquivalenceClassProtocol