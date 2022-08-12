:orphan:

.. _graph_api:

***
DAG
***

All basic causal graphs stem from the ``DAG`` class, so feel free to refer to basic 
graph operations based on the ``DAG`` class. Other more complicated graph structures,
such as the CPDAG, ADMG, or PAG have the same API and structure, but add on
different types of edges.

Overview
********
.. currentmodule:: dodiscover.graphs.base

All causal graphs are subclassed from the following abstract classes. These
are provided, so that users may extend the graph classes.

.. autosummary::
   :toctree: generated

   MarkovianGraph
   SemiMarkovianGraph
   MarkovEquivalenceClass

.. currentmodule:: dodiscover.graphs.dag

Copying
-------
.. autosummary::
   :toctree: generated

   DAG.copy
   DAG.to_adjacency_graph

Information about nodes and edges
---------------------------------
.. autosummary::
   :toctree: generated

   DAG.name
   DAG.parents
   DAG.children
   DAG.successors
   DAG.predecessors
   DAG.edges
   DAG.nodes
   DAG.has_adjacency
   DAG.has_edge
   DAG.has_node
   DAG.number_of_edges
   DAG.number_of_nodes
   DAG.size
   DAG.degree
   DAG.markov_blanket_of

Graph modification
------------------
.. autosummary::
   :toctree: generated

   DAG.add_node
   DAG.add_nodes_from
   DAG.remove_node
   DAG.remove_nodes_from
   DAG.remove_edge
   DAG.remove_edges_from
   DAG.add_edge
   DAG.add_edges_from
   DAG.clear
   DAG.clear_edges

Ordering
--------
.. autosummary::
   :toctree: generated

   DAG.order

Conversion to/from other formats
--------------------------------
.. autosummary::
   :toctree: generated

   DAG.to_dot_graph
   DAG.to_networkx
   DAG.to_numpy
   DAG.save