from typing import Optional, List, Text

import tensorflow as tf
import tf_agents as tfa

from .utils import types as ts
from .utils import keys
from .nodes import Node, InfoNode
from .policy import InfoNodePolicy


class InfoNodeAgent(tfa.agents.TFAgent):
    """InfoNodeAgent class

    This class performs the following roles:
    * Builds an `InfoNodePolicy` from nnn
    * Trains its policy on experience trajectories collected

    Its primary methods are:
    * train
    """

    def __init__(
            self,
            nodes: List[Node],
            observation_keys_nest: ts.NestedText,
            action_keys_nest: ts.NestedText,
            env_time_step_spec: ts.TimeStep,
            env_action_spec: ts.NestedTensorSpec,
            predix_id: Optional[str] = ''):
        """Initializes an `InfoNodeAgent`.

        Args:
            nodes: All the info_node_names to use in the agent's policy.
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

    def _train(
            self,
            experience: ts.NestedTensor,
            weights: Optional[tf.Tensor] = None,
            info_node_names: List[Text] = None,
            only_train_top_k: int = 0
            ) -> tfa.agents.tf_agent.LossInfo:
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
            info_node_names: Names of info_node_names to sequentially train. If None, this function trains
                all info_node_names sequentially. An empty list, however, trains *no* `InfoNode`s.
            only_train_top_k: If not 0, this parameter identifies How many lowest performing `InfoNode`s
                to select from `info_node_names` for training.

        Returns:
            A `LossInfo` containing the loss *before* the training step is taken.
            In most cases, if `weights` is provided, the entries of this tuple will
            have been calculated with the weights.  Note that each Agent chooses
            its own method of applying weights.
        """

        if info_node_names is None:
            info_node_names = self.info_node_policy.info_node_names

        prev_loss = self._loss(experience=experience, weights=weights, info_node_names=info_node_names)
        info_node_losses = prev_loss.extra  # structured as: {info_node.name: scalar_loss for info_node in nnn}

        if only_train_top_k != 0:
            k = min(only_train_top_k, len(info_node_losses))
            losses_ks = list(info_node_losses.keys())
            losses_vs = list(info_node_losses.values())
            info_node_losses_tensor = tf.constant(losses_vs)
            _, top_k_indeces = tf.math.top_k(info_node_losses_tensor, k=k)
            info_node_names = [losses_ks[i] for i in top_k_indeces]

        for node_name in info_node_names:
            node = self.info_node_policy.all_nodes[node_name]
            node.train(experience)

        return prev_loss

    def _loss(
            self,
            experience: ts.NestedTensor,
            weights: ts.Tensor,
            info_node_names: List[Text] = None,
            ) -> Optional[tfa.agents.tf_agent.LossInfo]:
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
            info_node_names: Names of `InfoNode`s to compute loss for. If None, this function evaluates
                the loss for all info_node_names. An empty list, however, returns loss for *no* nodes.

        Returns:
            A `LossInfo` containing the loss *before* the training step is taken.
            In most cases, if `weights` is provided, the entries of this tuple will
            have been calculated with the weights. The `extra` member of the `LossInfo`
            gives a per-infonode breakdown of loss.
        """

        if weights is None:
            batch_size = experience[self.info_node_policy.info_node_names[0]][keys.STATES.ENERGY].shape[0]
            weights = tf.ones((batch_size,))
        if info_node_names is None:
            info_node_names = self.info_node_policy.info_node_names

        node_loss = {info_node_name: tf.reduce_sum(weights*tf.reduce_sum(
                        node_experience[keys.STATES.ENERGY], axis=1), axis=0)
                     for info_node_name, node_experience
                     in experience.items()
                     if info_node_name in info_node_names}
        total_loss = sum(node_loss.values())

        return ts.LossInfo(loss=total_loss, extra=node_loss)