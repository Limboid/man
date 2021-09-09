from typing import Mapping, Text

from ..utils import types as ts
from . import Node


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
        self.f_bottom_up = f_bottom_up
        self.f_top_down = f_top_down

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return self.f_bottom_up(states)

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return self.f_top_down(states)