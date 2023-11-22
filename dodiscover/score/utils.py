import itertools
from itertools import combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def is_clique(G, nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n * (n - 1) / 2
