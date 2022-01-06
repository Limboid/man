import re
from typing import List, Tuple, Union, TypeVar, Optional

T = TypeVar('T')

_ALL_NAMES = list()
def ensure_unique(identifier: str) -> str:
    """Makes sure the same `identifier` is not declared twice. Non-idempotent.
    If this function is called on an `identifier` more than once, sucessive calls
    will result in an incremental postfix applied to `identifier`.

    Note:
        This function only preserves uniqueness across a single runtime environment.

    Example:
        >>>ensure_unique('Node')
        >>>'Node'
        >>>ensure_unique('Node')
        >>>'Node1'
        >>>ensure_unique('Node')
        >>>'Node2'
        >>>ensure_unique('Node1')
        >>>'Node3'
        >>>ensure_unique('Node4')
        >>>'Node4'

    Args:
        identifier: The string identifier to sanitize for uniqueness.

    Returns:
         Returns a unique form of `identifier`.
    """

    # see if identifier already exists
    if identifier not in _ALL_NAMES:
        _ALL_NAMES.append(identifier)
        return identifier

    # Gather information to change identifier:
    # get trailing numbers
    digits = []
    for c in reversed(identifier):
        if c.isnumeric():
            digits.append(c)
        else:
            break

    # partition identifier into the leading root and
    # trailing previous digits
    root = identifier[:len(digits)]
    prev_num = int(sum(digits)) if len(digits) > 0 else 0

    # call recursively with successive trailing numbers
    # until a unique identifier is found
    return ensure_unique(f'{root}{prev_num+1}')


def DONT_USE_check_all_same_type(L: List[T]):
    """THIS FUNCTION DOES NOT WORK
    IT ONLY WORKS IF EVERY ELEMENT EXACTLY MATCHES WITH NO SUBCLASSES

    Makes sure all elements in a list are of the same type.

    Args:
        L: list of items to check types equivalence

    Returns:
        Returns nothing if they match. Otherwise raises an error.
    """
    if len(L) <= 1:
        return

    for li in L[1:]:
        if type(li) != type(L[0]):
            raise 'Types do not match'

    return


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
        else:
            ret_lists.append(length * [list_or_var])

    return tuple(ret_lists)