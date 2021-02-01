from .node import Node


class LambdaNode(Node):
    """QT Node to use as subnodes in 'sensory' nodes. Not trainable.
    To make a trainable node, it's better to just subclass `InfoNode` yourself."""

    def __init__(self,
                 f_bottom_up,
                 f_top_down,
                 state_spec,
                 name='LambdaNode',
                 subnodes=None):
        super(LambdaNode, self).__init__(state_spec=state_spec, name=name, subnodes=subnodes)
        self.__setattr__('bottom_up', f_bottom_up)
        self.__setattr__('top_down', f_top_down)