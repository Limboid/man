"""semantics of functions here are explained in `AttnNode`'s docstring."""

from typing import Optional, Text, List, Mapping, Callable, Tuple, Union, Counter
from functools import partial

import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import convert_data_format
import tf_agents.typing.types as ts

from ...utils import keys, sanitize

keras = tf.keras


def f_bu_attn_LP_diff_factory(p: int):

    def fun(input: ts.NestedTensor) -> ts.NestedTensor:
        return tf.nest.map_structure(lambda x: tf.norm(x, ord=p, axis=-1), input)

    return fun


def f_td_attn_conv_similarity_factory(
    ndim: int,
    strides: Optional[Union[ts.Int, List[ts.Int]]] = 1,
    padding: Optional[Union[Text, List[Text]]] = None):
    """Perform similarity based searching with N-d convoultional matching.
    Inputs must have first (batch) and last (feature) axes.
    However these may be length 1. The bias must be shaped as
    (Batch, <Spatial shape>, features).

    Args:
        ndim: number of spatiotemperal dimensions to convolve over
        strides: keras style strides list or int. If not set to 1, you will need to
            upscale the output or impliment custom top_down attention calculation.
        padding: 'SAME' (default) or 'VALID'. If `'VALID'`, you will need to
            pad the output or impliment custom top_down attention calculation."""

    if padding is None:
        padding = 'SAME'

    def convolutional_similarity_single(input_i: tf.Tensor, bias_i: tf.Tensor):
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

    def fun(input: ts.NestedTensor, bias: ts.NestedTensor) -> ts.NestedTensor:
        return tf.nest.map_structure(convolutional_similarity_single, input, bias)

    return fun


def f_td_attn_dotprod_similarity(input: ts.NestedTensor, bias: ts.NestedTensor) -> ts.NestedTensor:

    def dotprod_similarity(input_i: tf.Tensor, bias_i: tf.Tensor):
        # input_i: [B, ..., F], attn_map_i: [B, ...].
        # `...` means spatiotemperal axes to compute attention over
        activation = tf.nn.relu(tf.einsum('...i,...j->...', input_i, bias_i)) # [B, N]
        return tf.expand_dims(activation, axis=-1) # [B, N, 1]

    return tf.nest.map_structure(dotprod_similarity, input, bias)


def f_td_attn_Lp_similarity_factory(p: int):

    def Lp_similarity(input_i: tf.Tensor, bias_i: tf.Tensor):
        return tf.norm(input_i - bias_i, ord=p, axis=-1, keepdims=False)

    def _f_td_attn_Lp_similarity(input: ts.NestedTensor, bias: ts.NestedTensor) -> ts.NestedTensor:
        return tf.nest.map_structure(Lp_similarity, input, bias)

    return _f_td_attn_Lp_similarity


def f_td_attn_ifft(input: ts.NestedTensor, bias: ts.NestedTensor) -> ts.NestedTensor:
    """each 'atomic' tensor in `bias` is a float32 dense [B, N, C, 2] tensor where B=batch size,
     N=number of dimensions, C=fourier coefficents, and 2=(amplitude, ofset). An attention map
     is computed by the inverse fourier transform of `bias` and returned as origonally nested """
    pass


def f_attend_softmax_pool(input: ts.NestedTensor, attention_map: ts.NestedTensor) -> ts.NestedTensor:

    def tensor_softmax_attend(input_i, attn_map_i):
        # input_i: [B, ..., F], attn_map_i: [B, ...].
        # `...` means spatiotemperal axes to compute attention over
        input_i = tf.reshape(input_i, shape=(input_i.shape[0], -1, input_i.shape[-1])) # [B, N, F]
        attn_map_i = tf.reshape(attn_map_i, shape=(attn_map_i.shape[0], -1)) # [B, N]
        attn_map_i = tf.math.softmax(attn_map_i, axis=-1) # [B, N]
        attended = attn_map_i * input_i # [B, N, F]
        return tf.reduce_sum(attended, axis=-2) # [B, F]

    return tf.nest.map_structure(tensor_softmax_attend, input, attention_map)
