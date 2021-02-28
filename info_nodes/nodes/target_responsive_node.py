from typing import Optional, Text, Mapping

from . import InfoNode
from .info_node import functions
from ..utils import keys
from ..utils import types as ts


class TargetResponsiveNode(InfoNode):
    """used for simple action nodes like SoftMovingGridAttnNode"""

    def __init__(self,
                 latent_spec: ts.NestedTensorSpec,
                 name: Optional[Text] = 'TargetResponsiveNode'):

        super(TargetResponsiveNode, self).__init__(
            state_spec_extras=dict(),
            parent_names=[],
            latent_spec=latent_spec,
            f_parent=functions.f_parent_no_parents,
            f_child=functions.f_child_sample_factory(0),
            subnodes=[],
            name=name)

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        energy, latent = self.f_child(targets=states[self.name][keys.STATES.TARGET_LATENTS])
        states[self.name][keys.STATES.ENERGY] = energy
        states[self.name][keys.STATES.LATENT] = latent
        return states

    def train(self, experience: ts.NestedTensor) -> None:
        pass