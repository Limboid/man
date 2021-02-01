from typing import Optional, List, Mapping, Text, Callable
from functools import reduce

import tensorflow as tf
import tf_agents as tfa
import tf_agents.typing.types as ts
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

from .nodes import Node, InfoNode


class InfoNodePolicy(tfa.policies.tf_policy.TFPolicy):

    def __init__(self,
                 nodes: List[Node],
                 observation_keys_nest: ts.Nested[Text],
                 action_keys_nest: ts.Nested[Text],
                 env_time_step_spec: ts.TimeStep,
                 env_action_spec: ts.NestedTensorSpec,
                 name: Optional[str] = '',
                 ):
        """Initializes an `InfoNodePolicy`.

        Args:
            nodes: All the node_names to use in the policy.
            observation_keys_nest: State keys to assign to the environment observation at the start of `_action`.
            action_keys_nest: Structure to give to values before returning action from the end of `_action`.
            env_time_step_spec: A nest of tf.TypeSpec representing the time_steps. Taken from the environment's
                `time_step_spec`.
            env_action_spec: A nest of BoundedTensorSpec representing the actions. Taken from the environment's
                `time_step_spec`.
            name: Prefix to use when naming variables in tensorflow backend.
        """

        """convert `node_names` into a flattened dictionary of all node_names that will be called during
        training/inference using their unique names for the keys."""
        self.all_nodes = reduce(
            (lambda flattened_nodes, node: flattened_nodes+[node]+node.subnodes),
            nodes, [])

        self._observation_keys_nest = observation_keys_nest
        self._action_keys_nest = action_keys_nest

        state_spec = {node.name: node.state_spec
                      for node in self.all_nodes}

        info_spec = {infonode.name: infonode.info_spec()
                     for infonode
                     in self.all_nodes
                     if isinstance(infonode, InfoNode)}

        super(InfoNodePolicy, self).__init__(
            time_step_spec=env_time_step_spec,
            action_spec=env_action_spec,
            policy_state_spec=state_spec,
            info_spec=info_spec,
            name=name)

    def _action(self,
                time_step: ts.TimeStep,
                policy_state: ts.NestedTensor,
                seed: Optional[ts.Seed]
                ) -> tfa.policies.tf_policy.policy_step.PolicyStep:
        """

        Args:
            time_step: TimeStep named tupled containing step_type, reward, info, observation
            policy_state: NestedTensor containing the
            seed:

        Returns:
            PolicyStep named tuple ('action', 'state', 'info')
        """
        states = policy_state

        # extract obs from time_step according to self._observation_keys_nest then put in keys in state
        obs_keys = tf.nest.flatten(self._observation_keys_nest)
        obs_vals = tf.nest.flatten(time_step.observation)
        obs_dict = {name: obs
                    for name, obs
                    in zip(obs_keys, obs_vals)}
        states.update(obs_dict)

        # node-wise bottom up
        # determine order
        for node in self.all_nodes:
            states = node.bottom_up(states)

        # I can still pass distributions between bottom up and top down, just not the other way around.

        # node-wise top down
        for node in reversed(self.all_nodes):
            states = node.top_down(states)

        # extract action values from state and structure according to self._action_keys_nest
        act_keys = tf.nest.flatten(self._action_keys_nest)
        act_vals = [states[k] for k in act_keys]
        action = tf.nest.pack_sequence_as(structure=act_keys, flat_sequence=act_vals)

        # get info for training
        info = self._get_info(states)

        return tfa.policies.tf_policy.policy_step.PolicyStep(action=action, state=states, info=info)

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        raise NotImplementedError('This policy does not support distribution output')

    def _get_info(self, states):
        losses = [infonode.loss(states)
                  for infonode
                  in self.all_nodes
                  if isinstance(infonode, InfoNode)]
        composite_loss_val = sum(loss_i.loss for loss_i in losses)
        composite_loss_extra = {loss.extra['name']: loss.extra for loss in losses}
        composite_loss = tfa.agents.tf_agent.LossInfo(loss=composite_loss_val, extra=composite_loss_extra)
        return dict(composite_loss=composite_loss, states=states)

    def _get_initial_state(self,
                           batch_size: Optional[ts.Int] = None
                           ) -> ts.NestedTensor:
        """Gets initial state for interaction with env.

        NOTE this method is meant to be called by users outside the class

        example:

        ```python
        env = SomeTFEnvironment()
        policy = TFRandomPolicy(env.time_step_spec(), env.action_spec())
        # Or policy = agent.policy or agent.collect_policy
        policy_state = policy.get_initial_state(env.batch_size)
        time_step = env.reset()
        while not time_step.is_last():
            policy_step = policy.action(time_step, policy_state)
            time_step = env.step(policy_step.action)
            policy_state = policy_step.state
            # policy_step.info may contain side info for logging, such as action log
            # probabilities.
        ```

        Args:
            batch_size: Tensor or constant: size of the batch dimension. Can be None
                in which case no dimensions gets added.

        Returns:
            `Nested` structure (possibly `Tensor`s) to initialize this `Node`'s state with
                during training/inference
        """
        return {node.name: node.initial_state(batch_size)
                for node in self.all_nodes}

    @property
    def node_names(self):
        return [node.name for node in self.all_nodes]