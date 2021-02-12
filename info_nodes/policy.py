from typing import Optional, List, Mapping, Text
from functools import reduce

import tensorflow as tf
import tf_agents as tfa

from .utils import types as ts
from .utils import keys
from .nodes import Node, InfoNode
from .nodes._env_interface import ObsInfoNode, ActInfoNode


class InfoNodePolicy(tfa.policies.tf_policy.TFPolicy):

    def __init__(
            self,
            nodes: List[Node],
            observation_keys_nest: ts.Nested[Text],
            action_keys_nest: ts.Nested[Text],
            env_time_step_spec: ts.TimeStep,
            env_action_spec: ts.NestedTensorSpec,
            name: Optional[str] = ''):
        """Initializes an `InfoNodePolicy`.

        Args:
            nodes: All the info_node_names to use in the policy.
            observation_keys_nest: State keys to assign to the environment observation at the start of `_action`.
            action_keys_nest: Structure to give to values before returning action from the end of `_action`.
            env_time_step_spec: A nest of tf.TypeSpec representing the time_steps. Taken from the environment's
                `time_step_spec`.
            env_action_spec: A nest of BoundedTensorSpec representing the actions. Taken from the environment's
                `time_step_spec`.
            name: Prefix to use when naming variables in tensorflow backend.
        """

        self.all_nodes = reduce(
            (lambda flattened_nodes, node: flattened_nodes + [node] + node.subnodes),
            nodes, [])

        self._observation_keys_nest = observation_keys_nest
        self._action_keys_nest = action_keys_nest

        for node in self.all_nodes:
            if isinstance(node, InfoNode):
                node.build(self.all_nodes)

        # make special InfoNodes to inject to/from latent states
        obs_nodes = []
        for key, obs_n in zip(tf.nest.flatten(observation_keys_nest),
                              tf.nest.flatten(env_time_step_spec.observation)):
            obs_nodes.append(ObsInfoNode(key=key, sample_observation=obs_n, all_nodes=self.all_nodes))

        act_nodes = []
        for key, act_n in zip(tf.nest.flatten(action_keys_nest),
                              tf.nest.flatten(env_action_spec)):
            act_nodes.append(ActInfoNode(key=key, sample_action=act_n, all_nodes=self.all_nodes))

        self.all_nodes = obs_nodes + act_nodes + self.all_nodes

        state_spec = {node.name: node.state_spec
                      for node in self.all_nodes}
        info_spec = self.info_spec

        super(InfoNodePolicy, self).__init__(
            time_step_spec=env_time_step_spec,
            action_spec=env_action_spec,
            policy_state_spec=state_spec,
            info_spec=info_spec,
            name=name)

    def _action(
            self,
            time_step: ts.TimeStep,
            policy_state: ts.NestedTensor,
            seed: Optional[ts.Seed]
            ) -> tfa.trajectories.policy_step.PolicyStep:
        """

        Args:
            time_step: TimeStep named tupled containing step_type, discount, reward, observation
            policy_state: NestedTensor containing the
            seed:

        Returns:
            PolicyStep namedtuple ('action', 'state', 'info')
        """
        states = policy_state

        # extract obs from time_step according to self._observation_keys_nest then put in keys in state
        for key, val in zip(tf.nest.flatten(self._observation_keys_nest), tf.nest.flatten(time_step.observation)):
            states[key][keys.STATES.LATENT] = val

        # node-wise bottom up
        for node in self.all_nodes:
            states = node.bottom_up(states)

        # node-wise top down
        for node in reversed(self.all_nodes):
            states = node.top_down(states)

        # extract action values from state and structure according to self._action_keys_nest
        act_keys = tf.nest.flatten(self._action_keys_nest)
        act_vals = [states[k][keys.STATES.LATENT] for k in act_keys]
        action = tf.nest.pack_sequence_as(structure=act_keys, flat_sequence=act_vals)

        # get info for training
        info = self.get_info(states)

        return PolicyStep(action=action, state=states, info=info)

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        raise NotImplementedError('This policy does not support distribution output')

    def _get_initial_state(self, batch_size: Optional[ts.Int] = None) -> ts.NestedTensor:
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

    @property
    def info_node_names(self):
        return [node.name for node in self.all_nodes if isinstance(node, InfoNode)]

    @property
    def info_spec(self) -> types.NestedTensorSpec:
        """info is the same as `states` at any point in time.
        It is used to store recurrent information for training later"""
        return {infonode.name: infonode.state_spec
                for infonode
                in self.all_nodes
                if isinstance(infonode, InfoNode)}

    def get_info(self, states: Mapping[Text, ts.NestedTensor]):
        return states
