from typing import Mapping, Text

import tensorflow as tf

from ...utils import types as ts
from ..info_node import InfoNode


class ObsInfoNode(InfoNode):
    """wrapper for observations
    on `bottom_up`, an `ObsInfoNode` assigns the corresponding value from
    `states[obs_key]` to `states[self.name][keys.LATENT]`."""

    def __init__(self, key, sample_observation, all_nodes):
        state_spec = tf.nest.map_structure(lambda tensor: tf.TensorSpec.from_tensor(tensor), sample_observation)
        super(ObsInfoNode, self).__init__(
            state_spec_extras=dict(),
            controllable_latent_spec=[],
            parent_names=[],
            num_children=len([node for node in all_nodes
                              if isinstance(node, InfoNode) and key in node.parent_names]),
            latent_spec=state_spec,
            f_parent=lambda states, parent_names: None,
            f_child=lambda targets: None,
            subnodes=[],
            name=key)

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        """InfoNodePolicy manually alters `states[self.name][keys.STATES.LATENT]`."""
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def train(self, experience: ts.NestedTensor) -> None:
        pass
