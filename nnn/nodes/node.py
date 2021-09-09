from typing import List, Tuple, Any, Mapping

from nnn.connections import Connection
from nnn.connections.port import Port
from nnn.utils.sanitize import ensure_unique


class Node:

    def __init__(self,
                 ports: List[Port],
                 name: str = None):
        """Base initializer for `Node`.

        Note:
            Child `Node`'s should use this method to declare their port attributes.

        Args:
            ports: The ports that this node supports. Usually 'statically' known.
            name: Name of node.
        """
        if name is None:
            name = 'node'

        self.ports = ports
        self.name = ensure_unique(name)
        self.built = False

    def build(self,
              incoming_connections: Mapping[Port, List[Connection]],
              outgoing_connections: Mapping[Port, List[Connection]]):
        self.built = True

    def __call__(self,
                 incoming_values: Mapping[Port, Mapping[Connection, Any]]) \
            -> Mapping[Port, Mapping[Connection, Any]]:
        pass
