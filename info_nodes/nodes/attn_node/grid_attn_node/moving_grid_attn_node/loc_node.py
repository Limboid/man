import math
from typing import Mapping, List, Optional, Union, Callable, Text

import tensorflow as tf

from ... import InfoNode
from ....utils import keys
from ....utils import types as ts

class LocNode(InfoNode):
    """Manages a location in rectangular space encoded in binary
    as a [D, ceil(lg2(L))]-shaped tensor where D is the number
    of dimensions, and L is the length of the longest dimension.

    Its latent is structured [D, ceil(lg2(L))]

    During `top_down` biasing, `LocNode` properly interpolates the
    location proportional to the difference in their rectangular
    space representations.

    TODO: make location use velocity and acceleration."""

    def __init__(self,
                 grid_shape: ts.Shape):

        self.grid_shape = tf.constant(grid_shape)
        num_dimensions = self.grid_shape.shape[0]
        max_length = tf.reduce_max(self.grid_shape)
        self.coefs = 2. ** tf.range(max_length, dtype=tf.keras.backend.floatx())
        # a metrix of successive powers of two:
        # [[1, 2, 4, 8, ...] # for first grid spatial dimension
        #  [1, 2, 4, 8, ...] # for second grid spatial dimension
        #         .
        #         .
        #         .
        #  [1, 2, 4, 8, ...]] # for final grid spatial dimension

        super(LocNode, self).__init__(
            state_spec_extras=dict(),
            parent_names=[],
            latent_spec=tf.TensorSpec(self.grid_shape),
            name='LocNode')

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        energy, target = self.f_child(targets=states[self.name][keys.STATES.TARGET_LATENTS])

        old_loc = self.get_real_loc(states)
        target_loc = self.decode_loc(target)

        beta = 0.5 + tf.exp(energy)

        new_loc = (1. - beta) * old_loc + beta * target_loc
        new_loc = tf.clip_by_value(new_loc, 0, self.grid_shape)

        new_loc_base_2_encoded = self.encode_loc(new_loc)

        states[self.name][keys.STATES.LATENT] = new_loc_base_2_encoded

        return states

    def get_real_loc(self, states) -> ts.Tensor:
        return self.decode_loc(states[self.name][keys.STATES.LATENT])

    def decode_loc(self, base_two_valued: ts.Tensor) -> ts.Tensor:
        """converts [..., N] base-2 encoded tensor into
        [...] real-valued tensor.

        NOTE: being `base-2 encoded` simply means the tensor's
        values are multiplied by successive powers of two. The
        actual values may be floating point."""

        return base_two_valued @ self.coefs[:, tf.newaxis]

    def encode_loc(self, real_valued: ts.Tensor) -> ts.Tensor:
        """converts a [...] real-valued tensor into the equivalent
        [..., N] base-2 encoded tensor."""

        raise NotImplementedError('I need to make a square wave function')
        return tf.nn.sigmoid(math.pi * (2 * (real_valued[..., tf.newaxis] / self.grid_matrix[:, 0] - 1) - 1))
