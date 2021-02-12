import tensorflow as tf

from . import types as ts


def difference(A: ts.NestedTensor, B: ts.NestedTensor):
    """similarity between 2 equally structured nested tensors"""
    diff = tf.nest.map_structure(lambda a, b: a - b, A, B)
    sum_diff = sum(tf.nest.flatten(diff))
    return sum_diff

# I may have to define my own map_nested and flatten functions to operate
# equivalently on structured and atomic tensors
