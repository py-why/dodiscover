from typing import Dict

import networkx as nx
import numpy as np


def _sample_from_graph(
    G,
    top_sort_idx,
    rng: np.random.Generator,
    domain_id: int,
    ignored_nodes=None,
) -> Dict:
    """Private function to sample a single iid sample from a graph for all nodes.

    Returns
    -------
    nodes_sample : dict
        The sample per node.
    """
    nodes_sample = dict()

    for node_idx in top_sort_idx:
        # get all parents
        parents = G.parents(node_idx)

        # sample noise
        if "domain_gaussian_noise_function" in G.nodes[node_idx]:
            mean = G.nodes[node_idx]["domain_gaussian_noise_function"][domain_id]["mean"]
            std = G.nodes[node_idx]["domain_gaussian_noise_function"][domain_id]["std"]
        else:
            mean = G.nodes[node_idx]["gaussian_noise_function"]["mean"]
            std = G.nodes[node_idx]["gaussian_noise_function"]["std"]
        node_noise = rng.normal(loc=mean, scale=std)
        node_sample = 0

        # sample weight and edge function for each parent
        for parent in parents:
            if parent in ignored_nodes or parent == node_idx:
                continue
            if len(G.nodes[node_idx]["parent_functions"]) == 0:
                continue

            weight = G.nodes[node_idx]["parent_functions"][parent]["weight"]
            func = G.nodes[node_idx]["parent_functions"][parent]["func"]
            try:
                node_sample += weight * func(nodes_sample[parent])
            except Exception as e:
                print(node_idx, list(parents))
                raise e

        # set the node attribute "functions" to hold the weight and function wrt each parent
        node_sample += node_noise
        nodes_sample[node_idx] = node_sample
    return nodes_sample
