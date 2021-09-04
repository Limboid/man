import typing as _ty

import tensorflow as _tf
import tf_agents as _tfa
from tf_agents.typing import types as _types

# used when scalar-tensor or plain scalars are allowed
Int = _types.Int
Float = _types.Float

Seed = _types.Seed

Tensor = _tf.Tensor
TensorSpec = _tf.TensorSpec

Nested = _types.Nested
NestedTensor = _types.NestedTensor
NestedTensorSpec = _types.NestedTensorSpec
NestedText = Nested[_ty.Text, 'NestedText']

TimeStep = _tfa.trajectories.time_step.TimeStep
StepType = _tfa.trajectories.time_step.StepType
PolicyStep = _tfa.trajectories.policy_step.PolicyStep
LossInfo = _tfa.agents.tf_agent.LossInfo # namedtuple(["loss", "extra"])

scalar_tensor_spec = _tf.TensorSpec(shape=(), )

# to represent rectangular shape boundaries
Shape = _ty.Union[Tensor, TensorSpec, _tf.TensorShape]