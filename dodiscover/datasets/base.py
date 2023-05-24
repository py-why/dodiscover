from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import pywhy_graphs as pgraphs
from joblib import Parallel, delayed

from . import linear, multidomain


def sample_from_graph(
    G: nx.DiGraph,
    n_samples: int = 1000,
    n_jobs: Optional[int] = None,
    random_state=None,
    sample_func="linear",
    **sample_kwargs,
):
    """Sample a dataset from a linear Gaussian graph.

    Assumes the graph only consists of directed edges. It is on the roadmap to
    implement support for bidirected edges.

    Parameters
    ----------
    G : Graph
        A linear DAG from which to sample. Must have been set up with
        :func:`pywhy_graphs.functional.make_graph_linear_gaussian`.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    n_jobs : Optional[int], optional
        Number of jobs to run in parallel, by default None.
    random_state : int, optional
        Random seed, by default None.
    sample_func : str, optional
        The sampling function to use. Can be one of 'linear' or 'multidomain'.
        Defaults to 'linear'.
    **sample_kwargs
        Keyword arguments to pass to the sampling function.

    Returns
    -------
    data : pd.DataFrame of shape (n_samples, n_nodes)
        A pandas DataFrame with the iid samples.
    """
    if hasattr(G, "get_graphs"):
        directed_G = G.get_graphs("directed")
    else:
        directed_G = G

    if isinstance(G, nx.DiGraph):
        G = pgraphs.AugmentedGraph(incoming_directed_edges=G)

    if not nx.is_directed_acyclic_graph(directed_G):
        raise ValueError("The input graph must be a DAG.")
    if not G.graph.get("linear_gaussian", True):
        raise ValueError("The input graph must be a linear Gaussian graph.")

    rng = np.random.default_rng(random_state)

    # Create list of topologically sorted nodes
    top_sort_idx = list(nx.topological_sort(directed_G))

    if hasattr(G, "augmented_nodes"):
        top_sort_idx = [node for node in top_sort_idx if node not in G.augmented_nodes]
        ignored_nodes = G.augmented_nodes
    else:
        ignored_nodes = None

    if sample_func == "linear":
        sample_func = linear._sample_from_graph
    elif sample_func == "multidomain":
        sample_func = multidomain._sample_from_graph

    # Sample from graph
    if n_jobs == 1:
        data = []
        for _ in range(n_samples):
            node_samples = sample_func(
                G, top_sort_idx, rng, ignored_nodes=ignored_nodes, **sample_kwargs
            )
            data.append(node_samples)
        data = pd.DataFrame.from_records(data)
    else:
        out = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(sample_func)(G, top_sort_idx, rng, ignored_nodes=ignored_nodes, **sample_kwargs)
            for _ in range(n_samples)
        )
        data = pd.DataFrame.from_records(out)

    return data
