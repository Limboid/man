from typing import Mapping, Text

import tensorflow as tf

from ...utils import types as ts
from ...utils import keys
from ..info_node import InfoNode
from ..info_node import functions


class ActInfoNode(InfoNode):
    """wrapper for actions.
    on `top_down`, an `ActInfoNode` samples a larget from
    `states[self.name][keys.TARGET_LATENTS]` and assigns it
    to `states[act_key]`."""

    def __init__(self, key, sample_action, all_nodes):
        state_spec = tf.nest.map_structure(lambda tensor: tf.TensorSpec.from_tensor(tensor), sample_action)
        super(ActInfoNode, self).__init__(
            state_spec_extras=dict(),
            controllable_latent_spec=[],
            parent_names=[],
            num_children=len([node for node in all_nodes
                              if isinstance(node, InfoNode) and key in node.parent_names]),
            latent_spec=state_spec,
            f_parent=lambda states, parent_names: None,
            f_child=functions.f_child_sample_factory(0.),
            subnodes=[],
            name=key)

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        """InfoNodePolicy reads `states[self.name][keys.STATES.LATENT]` for the action."""
        energy, latent = self.f_child(targets=states[self.name][keys.STATES.TARGET_LATENTS])
        states[self.name][keys.STATES.ENERGY] = energy
        states[self.name][keys.STATES.LATENT] = latent
        return states

    def train(self, experience: ts.NestedTensor) -> None:
        pass
