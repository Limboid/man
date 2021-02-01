from typing import Optional, Text, List, Mapping

import tensorflow as tf
import tf_agents as tfa
import tf_agents.typing.types as ts

from .node import Node


class InfoNode(Node):
    """
    The `InfoNode` corresponds to a node in a directed Bayesian graph. While all `Node` objects
    can recurrently alter the `states` during `bottom_up` and `top_down`, `InfoNode` objects also
    expose learning functions `train` and `loss` as well as the state key `InfoNode.LOSS_K`

    While all `Node` objects may modify arbitrary properties of themselves and other node_names,
    the `InfoNode` also maintains two specific properties:
    * `LATENT_K`: `NestedTensor` containing its Bayesian-modeled latent variable.
    * `LOSS_K`: 0-dimensional `Tensor` containing its intrinsic loss. Used for weighting training replay.
    """

    LATENT_K = 'LATENT'
    LOSS_K = 'LOSS'

    def __init__(self,
        state_spec_dict: Mapping[str, ts.NestedTensorSpec],
        controllable_latent_mask: tf.Tensor,
        name: Optional[str] = 'InfoNode',
        subnodes: Optional[List[Node]] = None):
        """Meant to be called by subclass constructors.

        Args:
            state_spec: the dict of (potentially further nested) variables to associate with this `Node` during
                training/inference. Must contain values for keys `InfoNode.LATENT_K` and `InfoNode.LOSS_K`.
            name: node name to attempt to use for variable scoping.
            subnodes: `Node` objects that are owned by this node.
        """
        assert InfoNode.LATENT_K in state_spec_dict
        assert InfoNode.LOSS_K in state_spec_dict

        if subnodes is None:
            subnodes = list()
        super(InfoNode, self).__init__(
            state_spec=state_spec_dict,
            name=name,
            subnodes=subnodes
        )
        self._controllable_latent_mask = controllable_latent_mask

    def train(self, experience: ts.NestedTensor) -> None:
        """

        Args:
            experience:
        """
        raise NotImplementedError('subclasses should define their own training method')

    def loss(self, states: Mapping[Text, ts.Nested], **kwargs) -> tfa.agents.tf_agent.LossInfo:
        """Default loss function for individual nodes. Subclasses should not need to override this
        method. All losses are already defined in the Node's state under the key `InfoNode.LOSS_K`.
        However, subclasses may call this method to pass in `extra` information to the `LossInfo`
        object.

        Args:
            states: states of all nodes in the InfoNodePolicy
            **kwargs: optional extra data to insert in `extra`

        Returns:
`           LossInfo object containing loss and possibly extra information as well.
        """
        if not kwargs: kwargs = dict()
        return tfa.agents.tf_agent.LossInfo(loss=states[self.name][InfoNode.LOSS_K],
                                            extra=kwargs)

    # NEED TO PLUG THIS FUNCTION INTO EVERYTHING INSTEAD OF LOSS
    def info(self, states):
        """info is the same as loss"""
        return self.loss(states)

    def info_spec(self, **kwargs) -> ts.NestedTensorSpec:
        """info_spec function for info nodes. Subclasses should not need to override this
        method. The info_spec is already defined in `InfoNode`. However, subclasses may call
        this method to pass in `extra` TensorSpec key-value pairs. If a subclass has

        Args:
            **kwargs: optional extra key-value pairs to insert in `info_spec`

        Returns:
`           LossInfo object containing loss and possibly extra information as well.
        """
        if not kwargs: kwargs = dict()
        loss_spec = tf.TensorSpec(shape=(), name=f'{self.name}_loss_spec')
        info_spec = tfa.agents.tf_agent.LossInfo(loss=loss_spec, extra=kwargs)
        return info_spec

    @property
    def controllable_latent_mask(self):
        return self._controllable_latent_mask
