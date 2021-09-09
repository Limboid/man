"""Various ops for data structure manipulation."""

from typing import List, Optional, Callable, Mapping, Iterable

from nnn.utils.types import Nested, NestedTensor, FloatLike, Tensor


def map_nested(func, *nests, **kwargs):
    """Maps a function `func` over nested structures (combinations of
    `Iterable` and `Mapping`). Preserves the origonal ordering / key-value
     assignment of the input data structures.

     Note:
         This function might not raise an exception if the data structures
         are not shaped equivalently.

     Example:
         >>>map_nested(sum, [1, 2, 3, 4], [0, -2, 0, -4])
         >>>[1, 0, 3, 0]

     Args:
         func: the function to map data from *nests to the output.
         *nests: data structures to map from.
         **kwargs: optional keyword arguments to `func`.

     Returns:
         returns the equivalently shaped data structure with elements
         mapped by `func`.
         """
    if isinstance(nests[0], Iterable):
        return [map_nested(func, *nest_tuple_i, **kwargs)
                for nest_tuple_i in zip(*nests)]
    elif isinstance(nests[0], Mapping):
        return {k: map_nested(func, *[nest[k] for nest in nests], **kwargs)
                for k in list(nests[0].keys())}
    else:
        return func(*nests, **kwargs)
        # OLD: raise '`map_nested` can only map over `list` and `dict` objects'

def flatten(nest: Nested) -> list:
    """Flattens the orphan items of nested structures (combinations of `Iterable`
    and `Mapping`) into a `list` in the order that they appear in their containers.
    Duplicates are allowed.

    Example:
        >>>flatten(sum, {1:[1, [2, 3], 4], 0:[0, -2, 0, -4]})
        >>>[0, -2, 0, -4, 1, 2, 3, 4]

    Args:
        nest: Nested data structure to flatten.

    Returns:
        Returns a list of the bottom-level items from `nest`.
        """
    if isinstance(nest, Iterable):
        return sum(flatten(v) for v in nest)
    elif isinstance(nest, Mapping):
        return sum(flatten(v) for v in list(nest.values()))
    else:
        return [nest]

def filter_dict(original: Mapping, keys: List[str]):
    """Only returns the elements in `original` that are specified by `keys`.

    Args:
        original: Mapping to filter
        """
    return {k: v for k, v in original.items() if k in keys}


def scalar_difference(A: NestedTensor, B: NestedTensor,
                      diff_func: Optional[Callable[[Tensor, Tensor], FloatLike]] = None):
    """Similarity between 2 equally structured nested tensors.

    Note:
        You should supply your own `diff_func` if you are working with nests of
        non-scalar data.

    Args:
        A: The first NestedTensor to compare.
        B: The second NestedTensor to compare.
        diff_func: Function to evaluate the difference between two tensors. Specifically,
        this function recieves A and B as input and evaluates their scalar difference.

    Returns:
        Returns the overall difference between A and B.
    """
    if diff_func is None:
        def diff_func(a,b):
            return a-b

    diff = map_nested(diff_func, A, B)
    sum_diff = sum(flatten(diff))
    return sum_diff