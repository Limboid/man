import typing as _typing

FloatLike = _typing.TypeVar('FloatLike')  # any float scalar (e.g.: `float` or `Tensor`)

# modified from tf-agents
# https://github.com/tensorflow/agents/blob/master/tf_agents/typing/types.py
Tnest = _typing.TypeVar('Tnest')
Trecursive = _typing.TypeVar('Trecursive')
Nested = _typing.Union[Tnest,
                       _typing.Iterable[Trecursive],
                       _typing.Mapping[str, Trecursive]]
NestedText = Nested[_typing.Text, 'NestedText']

Tensor = _typing.TypeVar('Tensor')
Vector = Tensor
NestedTensor = Nested[Tensor, 'NestedTensor']
NestedTensorShape = Nested[Vector, 'NestedTensorShape']