from typing import List, Tuple, Union, TypeVar, Optional

T = TypeVar

def make_list_same_len(*list_or_vars: Union[T, List[T]], assert_len: Optional[int] = None):
    """Makes all `list_or_var` inputs into enqually sized lists.
    Useful when you have multiple parameters that should all be the
    same length.

    Example:

    >>> kernel_size = 3
    >>> strides = [4,4]
    >>> make_list_same_len(kernel_size, strides)
    >>> [3,3], [4,4]

    Args:
        *list_or_vars: lists or single variables that should be copied into lists
        assert_len: optional to assert all lists are equal length. Also can give
            hint in the case no ordered parameters are lists.

    Returns:
        tuple of equally sized lists in the order supplied.
    """
    length = None
    for list_or_var in list_or_vars:
        if isinstance(list_or_var, (List, Tuple)):
            length_i = len(list_or_var)
            if length is None:
                length = length_i
            if length is not None:
                assert length == length_i, 'Not all of the parameters have matching lengths'

    if assert_len is not None and length is not None:
        assert assert_len == length, 'The parameters all have the same length,' + \
                                     'but it is not the same as the asserted length'
    elif assert_len is None and length is not None:
        pass
    elif assert_len is not None and length is None:
        length = assert_len
    elif assert_len is None and length is None:
        raise AttributeError('None of the parameters have a specified or implied length')

    ret_lists = list()
    for list_or_var in list_or_vars:
        if isinstance(list_or_var, List):
            ret_lists.append(list_or_var)
        elif isinstance(list_or_var, Tuple):
            ret_lists.append(list(list_or_var))
        else
            ret_lists.append(length * [list_or_var])

    return tuple(ret_lists)