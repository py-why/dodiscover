import networkx as nx 


def var_process_from_graph(graph: nx.DiGraph, n_times:int=1000):
    # NOTE: currently this is just a VAR(1) process
    adj_mat = nx.to_numpy_array(graph)

    