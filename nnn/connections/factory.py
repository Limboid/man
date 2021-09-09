import itertools
from typing import List, Callable

from nnn.connections import Connection
from nnn.nodes import Node


def make_chained_bottom_up_connections(nodes: List[Node]) -> List[Connection]:
    def f_connect(node_i, node_j, dist):
        if dist == 1:
            # node_j is exactly one step ahead of node_i
            return [Connection(node_i.top_port, node_j.bottom_port)]
        else:
            return []
    return make_connections_over_list(nodes, f_connect)

def make_chained_top_down_connections(nodes: List[Node]) -> List[Connection]:
    def f_connect(node_i, node_j, dist):
        if dist == -1:
            # node_j is exactly one step behind node_i
            return [Connection(node_i.bottom_port, node_j.top_port)]
        else:
            return []
    return make_connections_over_list(nodes, f_connect)

def make_fully_connected_side_connections(nodes: List[Node]) -> List[Connection]:
    def f_connect(node_i, node_j, dist):
        return [Connection(node_i.side_port, node_j.side_port)]
    return make_connections_over_list(nodes, f_connect)

def make_connections_over_list(
        nodes: List[Node],
        f_connect: Callable[[Node, Node, int], List[Connection]]) -> List[Connection]:
    """Generic connection builder over a list of nodes.

    Args:
        nodes: Input list of nodes.
        f_connect: A function that determines whether, what kind, and how many
            connections to form between all node pairs (including self-pairs) in
            the list. Its arguments are: `(node_i: Node, node_j: Node, dist: int)
            -> List[Connections]` for all nodes `node_i`, `node_j` in `nodes` and
            `dist = j - i`.

    Returns:
        Returns the list sum (repetitions are allowed) of all connections formed.
    """
    connections = []
    for (i, node_i), (j, node_j) in itertools.product(enumerate(nodes), enumerate(nodes)):
        connections += f_connect(node_i, node_j, j-i)
    return connections
