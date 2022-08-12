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

Most-used classes
=================
These are the causal classes for Structural Causal Models (SCMs), or various causal
graphs encountered in the literature. 

.. currentmodule:: dodiscover

.. autosummary::
   :toctree: generated/

   CPDAG
   ADMG

Converting Graphs
=================
.. currentmodule:: dodiscover.algorithms

.. autosummary::
   :toctree: generated/

   dag2cpdag
