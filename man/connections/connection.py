from man.connections.port import Port
from man.nodes.node import Node

class Connection:

    def __init__(self, src_port: Port, dst_port: Port):
        self.src_port = src_port
        self.dst_port = dst_port