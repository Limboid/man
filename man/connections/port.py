from man.utils.types import Nested
from man.utils.sanitize import ensure_unique


class Port:

    def __init__(self, port_type: str, shape: Nested = None):
        """Defines an interface between `Node`s.

        Args:
            port_type: The type of port that this port is.
            shape: Shape of data produced/consumed (optional).
        """
        self.port_type = port_type
        self.shape = shape


class BottomPort(Port):

    def __init__(self, shape: Nested = None):
        """Defines an interface between `Node`s.

        Args:
            shape: Shape of data produced/consumed (optional).
        """
        super(BottomPort, self).__init__('bottom_port', shape)


class TopPort(Port):

    def __init__(self, shape: Nested = None):
        """Defines an interface between `Node`s.

        Args:
            shape: Shape of data produced/consumed (optional).
        """
        super(TopPort, self).__init__('top_port', shape)


class SidePort(Port):

    def __init__(self, shape: Nested = None):
        """Defines an interface between `Node`s.

        Args:
            shape: Shape of data produced/consumed (optional).
        """
        super(SidePort, self).__init__('side_port', shape)