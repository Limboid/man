from typing import Optional, Text, List, Mapping, Callable, Tuple, Union, Counter
from functools import partial

import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import convert_data_format
import tf_agents.typing.types as ts

from ..utils import keys, sanitize
from . import info_node
from .info_node import InfoNode

keras = tf.keras


class AttentionInfoNode(InfoNode):
    """
    Generic AttentionInfoNode.

    This InfoNode is composed of modular attention functions:
    - f_parent
    - f_child
    - f_bu_attn
    - f_td_attn
    - f_attend
    Function implimentations are prefixed by their purpose name (EG: `f_attend_L1`)

    See subclasses for implementation-specifics of f_bu_attn, f_td_attn, and f_attend.
    """

    def __init__(self,
                 f_parent: Callable,
                 f_child: Callable,
                 f_bu_attn: Callable,
                 f_td_attn: Callable,
                 f_attend: Callable,
                 attention_value_spec: ts.NestedTensorSpec,
                 attention_bias_spec: ts.NestedTensorSpec,
                 parent_names: List[Text],
                 num_children: ts.Int,
                 name: str = 'AttentionInfoNode'):
        """meant to be called by subclasses"""

        latent_spec = {
            keys.STATES.ATTENTION_BIAS: attention_bias_spec,
            keys.STATES.ATTENTION_VALUE: attention_value_spec,
        }
        controllable_latent_spec = {
            keys.STATES.ATTENTION_BIAS: tf.nest.map_structure(lambda _: True, attention_bias_spec),
            keys.STATES.ATTENTION_VALUE: tf.nest.map_structure(lambda _: False, attention_value_spec),
        }

        super(AttentionInfoNode, self).__init__(
            state_spec_extras=dict(),
            controllable_latent_spec=controllable_latent_spec,
            parent_names=parent_names,
            num_children=num_children,
            latent_spec=latent_spec,
            name=name
        )

        self.f_parent = f_parent
        self.f_child = f_child
        self.f_bu_attn = f_bu_attn
        self.f_td_attn = f_td_attn
        self.f_attend = f_attend

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:

        # get attention information
        parent_energy, parent_latent = self.f_parent(states=states, parent_names=self.parent_names)
        child_energy, child_target_latent = self.f_child(targets=states[self.name][keys.TARGET_LATENTS])
        # perform attention
        bottom_up_attn_map = self.f_bu_attn(input=parent_latent)
        top_down_attn_map = self.f_td_attn(input=parent_latent, bias=child_target_latent)
        attention_map = tf.nest.map_structure(lambda bu, td: parent_energy*bu + child_energy*td,
                                              bottom_up_attn_map, top_down_attn_map)
        attended_value: ts.NestedTensor = self.f_attend(input=parent_latent, attention_map=attention_map)
        states[self.name][keys.STATES.LATENT][keys.STATES.ATTENTION_VALUE] = attended_value
        states[self.name][keys.STATES.LATENT][keys.STATES.ATTENTION_BIAS] = child_target_latent
        states[self.name][keys.STATES.ENERGY] = parent_energy + child_energy
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def train(self, experience: ts.NestedTensor) -> None:
        raise NotImplementedError()


class DenseSimilarityBasedAttentionInfoNode(AttentionInfoNode):
    """Attends to the last dimension of dense vectors by bottom up and top down processes:
    - Bottom up attention is determined by the L1 norm of the last dimension.
    - Top down attention is determined by the Lp similarity between a bias tensor
        and final dimension tensor slice."""


class Conv2DSimilarityBasedAttentionInfoNode(AttentionInfoNode):
    """Attends to the last dimension of dense vectors by bottom up and top down processes:
    - Bottom up attention is determined by the L1 norm of the last dimension.
    - Top down attention is determined by the L1 similarity between a bias tensor
        and convolutional windows over the input tensor."""


class NestedDenseAttentionInfoNode(AttentionInfoNode):
    """Attends to the last dimension of dense vectors by bottom up and top down processes:
    - Bottom up attention is determined by the L1 norm of the last dimension.
    - Top down attention is determined by an inverse fourier transform to locate
        a particular last dimension slice."""

    def __init__(self,
        parent_latent_spec: ts.NestedTensorSpec,
        parent_names: List[Text],
        num_children: int,
        f_parent: Callable,
        f_child: Callable,
        name='NestedDenseAttentionInfoNode'):

        bias_spec, value_spec = _f_attend_softmax_pool_value_spec(parent_latent_spec)
        super(NestedDenseAttentionInfoNode, self).__init__(
            f_bu_attn=_f_bu_attn_L1_diff,
            f_td_attn=_f_td_attn_ifft,
            f_attend=_f_attend_softmax_pool,
            attention_value_spec=value_spec,
            attention_bias_spec=bias_spec,
            parent_names=parent_names,
            num_children=num_children,
            f_parent=f_parent,
            f_child=f_child,
            name=name)


def _f_bu_attn_LP_diff_factory(p: int):
    def fun(input: ts.NestedTensor):
        return tf.nest.map_structure(lambda x: tf.norm(x, ord=p, axis=-1), input)
    return fun


def _f_td_attn_conv_similarity_factory(
    ndim: int,
    strides: Optional[Union[ts.Int, List[ts.Int]]] = 1,
    padding: Optional[Union[Text, List[Text]]] = None):
    """Perform similarity based searching with N-d convoultional matching.
    Inputs must have first (batch) and last (feature) axes.
    However these may be length 1. The bias must be shaped as
    (Batch, <Spatial shape>, features).

    Args:
        strides: keras style strides list or int. If not set to 1, you will need to
            upscale the output or impliment custom top_down attention calculation.
        padding: 'SAME' (default) or 'VALID'. If `'VALID'`, you will need to
            pad the output or impliment custom top_down attention calculation."""

    if padding is None:
        padding = 'SAME'

    def _convolutional_similarity_single(input_i: tf.Tensor, bias_i: tf.Tensor):
        # tf.nn.conv* like its filter shaped `spatial_dims + (channels_in, channels_out)`
        # EG: conv2d filters are shaped `(H, W, C, filters)`.
        # Since I only want 1 filter, I insert an axis at -1
        bias_i = tf.expand_dims(bias_i, axis=-1)
        conv_norm = tf.nn.convolution(input=input_i,
                                      filters=bias_i,
                                      strides=strides,
                                      padding=padding,
                                      data_format=convert_data_format('channels_last', ndim=ndim))
        return tf.abs(tf.squeeze(conv_norm, axis=-1))

    def fun(input: ts.NestedTensor, bias: ts.NestedTensor):
        return tf.nest.map_structure(_convolutional_similarity_single, input, bias)

    return fun

def _f_td_attn_dense_similarity(input: ts.NestedTensor, bias: ts.NestedTensor):
    def dense_similarity(input_i: tf.Tensor, bias_i: tf.Tensor):
        activation = tf.nn.relu(tf.einsum('...i,...j->...', input_i, bias_i))
        return tf.expand_dims(activation, axis=-1)
    return tf.nest.map_structure(dense_similarity, input, bias)

def _f_td_attn_L1_similarity(input: ts.NestedTensor, bias: ts.NestedTensor):
    def L1_similarity(input_i: tf.Tensor, bias_i: tf.Tensor):
        return tf.norm(input_i - bias_i, ord=1, axis=-1, keepdims=False)
    return tf.nest.map_structure(L1_similarity, input, bias)

def _f_td_attn_ifft(input: ts.NestedTensor, bias: ts.NestedTensor):
    """each 'atomic' tensor in `bias` is a float32 dense [B, N, C, 2] tensor where B=batch size,
     N=number of dimensions, C=fourier coefficents, and 2=(amplitude, ofset). An attention map
     is computed by the inverse fourier transform of `bias` and returned as origonally nested """
    pass

def _f_attend_softmax_pool_value_spec(input_spec: ts.NestedTensorSpec):
    bias_spec = TODO
    value_spec = tf.nest.map_structure(lambda tensor_spec: tensor_spec[0]+tensor_spec[-1], input_spec)
    return bias_spec, value_spec

def _f_attend_softmax_pool(input: ts.NestedTensor, attention_map: ts.NestedTensor):

    def tensor_softmax_attend(input_i, attn_map_i):
        input_i = tf.reshape(input_i, shape=(input_i.shape[0], -1, input_i.shape[-1]))
        attn_map_i = tf.reshape(attn_map_i, shape=(attn_map_i.shape[0], -1))
        attn_map_i = tf.math.softmax(attn_map_i, axis=-1)
        attended = attn_map_i * input_i # [B, N, X]
        return tf.reduce_sum(attended, axis=-2) # [B, X]

    return tf.nest.map_structure(tensor_softmax_attend, input, attention_map)
