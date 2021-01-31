from typing import Optional, List, Mapping, Text, Callable
from functools import reduce

import tensorflow as tf
import tf_agents as tfa
import tf_agents.typing.types as ts

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
        # self.all_nodes = { node.name: node for node in self.all_nodes }

        self._observation_keys_nest = observation_keys_nest
        self._action_keys_nest = action_keys_nest
        self.predix_id = name

        _state_spec = {node.name: node.state_spec
                       for node in self.all_nodes}

        super(InfoNodePolicy, self).__init__(
            time_step_spec=env_time_step_spec,
            action_spec=env_action_spec,
            policy_state_spec=_state_spec,
            name=name
        )

    def _action(self,
                time_step: ts.TimeStep,
                policy_state: ts.NestedTensor,  # I'll respect this for now, but I want to store Distributions
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
        assert isinstance(states, dict)

        # extract obs from time_step according to self._observation_keys_nest then put in keys in state
        obs_keys = tf.nest.flatten(self._observation_keys_nest)
        obs_vals = tf.nest.flatten(time_step.observation)
        obs_dict = {name: obs
                    for name, obs
                    in zip(obs_keys, obs_vals)}
        states.update(obs_dict)

        # node-wise bottom up
        # determine order
        priorities = [node.bottom_up_priority(states)
                      for node in self.all_nodes] # [N]-list<[B]-Tensor>
        priorities = tf.stack(priorities, axis=1) # [B,N]-Tensor
        bottom_up_fns = [node.bottom_up for node in self.all_nodes]
        self._execute_in_order(priorities=priorities, fns=bottom_up_fns, states=states)

        # I can still pass distributions between bottom up and top down, just not the other way around.

        # node-wise top down
        priorities = [node.top_down_priority(states)
                      for node in self.all_nodes] # [N]-list<[B]-Tensor>
        priorities = tf.stack(priorities, axis=1) # [B,N]-Tensor
        top_down_fns = [node.top_down for node in self.all_nodes]
        self._execute_in_order(priorities=priorities, fns=top_down_fns, states=states)

        # tabulate instrinsic loss
        losses = [infonode.loss(states)
                  for infonode
                  in self.all_nodes
                  if isinstance(infonode, InfoNode)]
        composite_loss_val = sum(loss.loss for loss in losses)
        composite_loss_extra = {loss.extra['name']: loss.extra for loss in losses}
        composite_loss = tfa.agents.tf_agent.LossInfo(loss=composite_loss_val, extra=composite_loss_extra)

        # extract action values from state and structure according to self._action_keys_nest
        act_keys = tf.nest.flatten(self._action_keys_nest)
        act_vals = [states[k] for k in act_keys]
        action = tf.nest.pack_sequence_as(structure=act_keys, flat_sequence=act_vals)

        # add arbitrary info for later debugging
        info = { "intrinsic_loss": composite_loss }

        return tfa.policies.tf_policy.policy_step.PolicyStep(action=action, state=states, info=info)

    def _execute_in_order(self,
                          priorities: tf.Tensor,
                          fns: List[Callable],
                          states: ts.NestedTensor,
                          call_batched: bool = False,
                          *fn_args,
                          **fn_kwargs
                          ) -> ts.NestedTensor:
        """Internal helper function for `_action`

        Args:
            priorities: 1D or 2D `Tensor` of orderings to execute functions in. Lower -> first; higher -> last.
                Identical values result in the earlier function in `fns` being executed.
            fns: List of functions to execute. These functions should all share the same signature.
                Each function must accept and return a `states` object.
            states: Recurrent nested data to pass through each function in the execution chain.
            call_batched: Determines if the orderings should be considered on an individual elementwise basis,
                or if only the mean across the batch should be used for ordering. During parallel inference, set
                `False` since one agent may have a wildly different state than another. During training on similar
                trajectories, set `True`. This will take advantage of parallelization and train faster. NOTE: when
                `True`, `ordering` and `states` must also include a batch dimension.
            *fn_args: Optional leading ordered arguments to call the functions with.
            **fn_kwargs: Optional keyword arguments to call the functions with.

        Returns:
            updated `states` `NestedTensor` object.
        """
        if not call_batched and priorities.shape[0] > 1:
            """convert [B,...] tensors into B-List<[1,...]-Tensor>"""
            def split_fn(x): return tf.split(x, num_or_size_splits=x.shape[0], axis=0)

            priorities_elems = split_fn(priorities)
            states_elems = tf.nest.map_structure(split_fn, states)
            states = [self._execute_in_order(priorities=ordering_elem, fns=fns, call_batched=True,
                                             states=states_elem, *fn_args, **fn_kwargs)
                      for ordering_elem, states_elem
                      in zip(priorities_elems, states_elems)]
            return tf.nest.map_structure(tf.stack, states)
        else:
            if call_batched and priorities.shape[0] > 1:
                priorities = tf.reduce_mean(priorities, axis=0, keepdims=True)

            # actually do the work here
            # order functions
            ordering = tf.argsort(priorities)
            ordered_fns = len(fns) * [(lambda states, *args, **kwargs: states)]  # initialize empty list
            for ordered_index, fn in zip(ordering, fns):
                ordered_fns[ordered_index] = fn
            # execute `ordered_fns` sequentially
            for fn in ordered_fns:
                states = fn(states=states, *fn_args, **fn_kwargs)
            return states

    """
    def _get_latent_spec(self) -> Mapping[Text, tf.TensorSpec]:
        ""gets nested TensorSpec of reward""
        return {name: node.state_spec[InfoNode.LATENT_K]
                for name, node
                in self.all_nodes
                if isinstance(node, InfoNode)}

    def _get_loss_spec(self) -> Mapping[Text, tf.TensorSpec]:
        ""gets nested TensorSpec of loss""
        return {name: node.state_spec[InfoNode.LOSS_K]
                for name, node
                in self.all_nodes
                if isinstance(node, InfoNode)}
    """

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
        return {name: node.initial_state(batch_size)
                for name, node
                in self.all_nodes.items()}

    @property
    def node_names(self):
        return list(self.all_nodes.keys())


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
            collect_policy=self.info_node_policy)

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