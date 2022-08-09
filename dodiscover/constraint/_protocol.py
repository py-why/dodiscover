from typing import Protocol


class GraphProtocol(Protocol):
    def nodes(self):
        pass

    def add_node(self, node_for_adding, **attr):
        pass

    def remove_node(self, u):
        pass

class EquivalenceClassProtocol(Protocol):
    