from typing import Text

from .. import TargetResponsiveNode
from ...utils import types as ts


class ActWrapperNode(TargetResponsiveNode):
    """wrapper for actions.
    on `top_down`, an `ActWrapperNode` samples a larget from
    `states[self.name][keys.TARGET_LATENTS]` and assigns it
    to `states[act_key]`.

    InfoNodePolicy reads `states[self.name][keys.STATES.LATENT]` for the action.
    """

    def __init__(self, key: Text, act_spec: ts.NestedTensorSpec):
        super(ActWrapperNode, self).__init__(latent_spec=act_spec, name=key)