from typing import Mapping, Text

from .. import InfoNode
from ...utils import types as ts
from ..info_node import functions


class ObsWrapperNode(InfoNode):
    """wrapper for observations
    on `bottom_up`, an `ObsWrapperNode` assigns the corresponding value from
    `states[obs_key]` to `states[self.name][keys.LATENT]`."""

    def __init__(self, key, obs_spec):
        super(ObsWrapperNode, self).__init__(
            state_spec_extras=dict(),
            parent_names=[],
            latent_spec=obs_spec,
            f_parent=functions.f_parent_no_parents,
            f_child=functions.f_child_no_children,
            subnodes=[],
            name=key)

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        """InfoNodePolicy manually alters the value at `states[self.name][keys.STATES.LATENT]`."""
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def train(self, experience: ts.NestedTensor) -> None:
        pass
