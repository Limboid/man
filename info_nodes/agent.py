from typing import Optional, List, Mapping, Text, Callable
from functools import reduce

import tensorflow as tf
import tf_agents as tfa
import tf_agents.typing.types as ts
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

from .nodes import Node, InfoNode
from .policy import InfoNodePolicy


class InfoNodeAgent(tfa.agents.TFAgent):
    """InfoNodeAgent class

    This class performs the following roles:
    * Builds an `InfoNodePolicy` from info_nodes
    * Trains its policy on experience trajectories collected

    Its primary methods are:
    * train
    """

    def __init__(self,
        nodes: List[Node],
        observation_keys_nest: ts.Nested[Text],
        action_keys_nest: ts.Nested[Text],
        env_time_step_spec: ts.TimeStep,
        env_action_spec: ts.NestedTensorSpec,
        predix_id: Optional[str] = ''):
        """Initializes an `InfoNodeAgent`.

        Args:
            nodes: All the node_names to use in the agent's policy.
            observation_keys_nest: State keys to assign to the environment observation at the start of `_action`.
            action_keys_nest: Structure to give to values before returning action from the end of `_action`.
            env_time_step_spec: A nest of tf.TypeSpec representing the time_steps. Taken from the environment's
                `time_step_spec`.
            env_action_spec: A nest of BoundedTensorSpec representing the actions. Taken from the environment's
                `time_step_spec`.
            predix_id: Prefix to use when naming variables in tensorflow backend.
        """
        self.nodes = nodes
        self.predix_id = predix_id

        self.info_node_policy = InfoNodePolicy(
            nodes=nodes,
            observation_keys_nest=observation_keys_nest,
            action_keys_nest=action_keys_nest,
            env_time_step_spec=env_time_step_spec,
            env_action_spec=env_action_spec,
            name=predix_id)

        super(InfoNodeAgent, self).__init__(
            time_step_spec=env_time_step_spec,
            action_spec=env_action_spec,
            policy=self.info_node_policy,
            collect_policy=self.info_node_policy,
            train_sequence_length=None)

    def _train(self,
               experience: ts.NestedTensor,
               weights: Optional[tf.Tensor] = None,
               node_names: List[Text] = None,
               ) -> ts.LossInfo:
        """Returns an op to train the agent.

        Args:
            experience: A batch of experience data in the form of a `Trajectory`. The
                structure of `experience` must match that of `self.training_data_spec`.
                All tensors in `experience` must be shaped `[batch, time, ...]` where
                `time` must be equal to `self.train_step_length` if that property is
                not `None`.
            weights: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
                containing weights to be used when calculating the total train loss.
                Weights are typically multiplied elementwise against the per-batch loss,
                but the implementation is up to the Agent.
            node_names: Names of node_names to sequentially train. If None, this function trains
                all node_names sequentially. An empty list, however, trains *no* node_names.

        Returns:
            A `LossInfo` containing the loss *before* the training step is taken.
            In most cases, if `weights` is provided, the entries of this tuple will
            have been calculated with the weights.  Note that each Agent chooses
            its own method of applying weights.
        """
        if node_names is None:
            node_names = self.info_node_policy.all_nodes.keys()
            node_names = list(node_names)

        prev_loss = self._loss(experience=experience, weights=weights, node_names=node_names)

        for node_name in node_names:
            node = self.info_node_policy.all_nodes[node_name]

            if isinstance(node, InfoNode):
                node.train(experience[node_name])

        return prev_loss


    def _loss(self,
              experience: ts.NestedTensor,
              weights: ts.Tensor,
              node_names: List[Text] = None,
              ) -> Optional[ts.LossInfo]:
        """Computes loss.
        This method does not increment self.train_step_counter or upgrade gradients.
        By default, any networks are called with `training=False`.
        Args:
            experience: A batch of experience data in the form of a `Trajectory`. The
                structure of `experience` must match that of `self.training_data_spec`.
                All tensors in `experience` must be shaped `[batch, time, ...]` where
                `time` must be equal to `self.train_step_length` if that property is not
                `None`.
            weights: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
                containing weights to be used when calculating the total train loss.
                Weights are typically multiplied elementwise against the per-batch loss,
                but the implementation is up to the Agent.
            node_names: Names of node_names to compute loss for. If None, this function evaluates
                the loss for all node_names. An empty list, however, returns loss for *no* node_names.

        Returns:
            A `LossInfo` containing the loss *before* the training step is taken.
            In most cases, if `weights` is provided, the entries of this tuple will
            have been calculated with the weights.  Note that each Agent chooses
            its own method of applying weights.
        """


        ####### TODO #######
        ## I modified the infonode.loss function ##
        ## actually run the trajectory
        steps = tf.split(expe, axis=1)
        for step in steps:
            do stuff()

        total_loss = 0
        loss_info = dict()
        with tf.name_scope(f'{self.name}_loss'):

            for node_name in node_names:
                node = self.info_node_policy.all_nodes[node_name]

                if isinstance(node, InfoNode):
                    node_loss = node.loss(experience[node_name])
                    total_loss = total_loss + node_loss
                    loss_info[node_name] = node_loss.extra

            if weights is not None:
                loss = tf.tensordot(total_loss, weights, axis=0, name='total_loss')
            return ts.LossInfo(loss, None)