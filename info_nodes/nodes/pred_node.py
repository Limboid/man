from typing import Optional, List, Text, Mapping

import tensorflow as tf
keras = tf.keras
import tf_agents.typing.types as ts

from .info_node import InfoNode
from ..utils import keys


class PredNode(InfoNode):

    def __init__(self,
        f_abs: keras.Model,
        f_pred: keras.Model,
        f_act: keras.Model,
        num_children: ts.Int,
        latent_spec: ts.NestedTensorSpec,
        name: Text = 'PredNode'):

        scalar_spec = tf.TensorSpec((1,))
        state_spec_extras = {
            keys.PRED_LATENT: latent_spec,
            keys.PRED_LATENT_UNCERTAINTY: scalar_spec
        }
        controllable_latent_mask = tf.nest.map_structure((lambda _: True), latent_spec)

        super(PredNode, self).__init__(
            state_spec_extras=state_spec_extras,
            controllable_latent_mask=controllable_latent_mask,
            latent_spec=latent_spec,
            num_children=num_children,
            subnodes=[],
            name=name
        )

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        pass

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        pass