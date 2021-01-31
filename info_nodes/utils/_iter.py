"""from typing import Mapping, List, Callable

from .types import Nested


def map_nested(map_fn: Callable[[any], any], nest: Nested):
    if isinstance(nest, List):
        return [map_nested(map_fn, i)
                for i in nest]
    elif isinstance(nest, Mapping):
        return [map_nested(map_fn, i)
                for i in nest]
    else:
        try: map_fn(nest)
        except ValueError as err:
            raise err
        except:
            raise NotImplementedError(f'`nest` {nest} is not a `typing.Mapping` or `typing.List`'
                                      f'and `map_fn` {map_fn} does not accept it')
"""