from typing import Mapping, Text, List, Optional, Callable

import tensorflow as tf

from . import types as ts


def prune(origonal: Mapping, keys: List[Text]):
    return {k: v for k, v in origonal.items() if k in keys}


def difference(A: ts.NestedTensor, B: ts.NestedTensor, diff_func: Optional[Callable] = None):
    """similarity between 2 equally structured nested tensors"""
    if diff_func is None:
        def diff_func(a,b):
            return a-b
    diff = tf.nest.map_structure(diff_func, A, B)
    sum_diff = sum(tf.nest.flatten(diff))
    return sum_diff


def flatten_time_into_batch_axis(experience: ts.NestedTensor):
    """Flattens the time and batch axes for nested tensors.
    For every atomic input tensor with dimensions `BT...`, the
    corresponding output tensor becomes `(B*T)...`.

    Args:
        experience: A nested tensor with all atomic tensors rank-2 or higher.

    Returns:
        A nested tensor with all atomic tensor ranks reduced by one.
    """
    def pack_single(tensor: tf.Tensor):
        new_shape = (-1,) + tensor.shape[2:]
        return tf.reshape(tensor, shape=new_shape)

    return tf.nest.map_structure(pack_single, experience)
