from collections import Mapping
from typing import Any, List

from nnn.connections import Connection
from nnn.connections.port import Port
from nnn.nodes import Node


class ARNode(Node):

    def __init__(self, model, name):
        super(ARNode, self).__init__(name=name)
        self.model = model
        self.bottom_port = Port()

    def build(self,
              incoming_connections: Mapping[Port, List[Connection]],
              outgoing_connections: Mapping[Port, List[Connection]]):
        self.built = True

    def __call__(self,
                 incoming_values: Mapping[Port, Mapping[Connection, Any]]) \
            -> Mapping[Port, Mapping[Connection, Any]]:
        pass
