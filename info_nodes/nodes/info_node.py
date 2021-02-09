from typing import Optional, Text, List, Mapping, Callable, Tuple

import tensorflow as tf
import tf_agents.typing.types as ts

from .node import Node
from ..utils import keys


class InfoNode(Node):
    """
    The `InfoNode` corresponds to a node in a directed Bayesian graph. While all `Node` objects
    can recurrently alter the `states` during `bottom_up` and `top_down`, `InfoNode` objects also
    expose learning functions `train` and `loss` as well as the state key `InfoNode.LOSS_K`

    While all `Node` objects may modify arbitrary properties of themselves and other info_node_names,
    the `InfoNode` also maintains two specific properties:
    * `LATENT`: `NestedTensor` containing its Bayesian-modeled latent variable.
    * `TARGET_LATENTS`: a list of 2-tuples of a 0-dimesnional `Tensor` and a `NestedTensor`s identifying
            probabilistic energy and value of latent set points.
    * `ENERGY`: probabilistic energy of latent (0-dimensional `Tensor`)
    * `ENERGY`: 0-dimensional `Tensor` containing its intrinsic loss. Used for weighting training replay.
    """

    def __init__(self,
                 state_spec_extras: Mapping[Text, ts.NestedTensorSpec],
                 controllable_latent_spec: Optional[ts.NestedTensor],
                 parent_names: List[Text],
                 num_children: ts.Int,
                 latent_spec: ts.NestedTensorSpec,
                 f_parent: Callable,
                 f_child: Callable,
                 subnodes: Optional[List[Node]] = None,
                 name: Optional[str] = 'InfoNode'):
        """Meant to be called by subclass constructors.

        Args:
            state_spec_dict: the dict of (potentially further nested) variables to associate with this `Node` during
                training/inference. Must contain values for keys `InfoNode.LATENT_K` and `InfoNode.LOSS_K`.
            subnodes: `Node` objects that are owned by this node.
            name: node name to attempt to use for variable scoping.
        """

        scalar_spec = tf.TensorSpec((1,))
        state_spec_dict = {
            keys.ENERGY: scalar_spec,
            keys.LATENT: latent_spec,
            keys.TARGET_LATENTS: num_children * [(
                scalar_spec, controllable_latent_spec
            )]
        }
        state_spec_dict.update(state_spec_extras)

        if subnodes is None:
            subnodes = list()

        super(InfoNode, self).__init__(
            state_spec=state_spec_dict,
            name=name,
            subnodes=subnodes
        )
        self._controllable_latent_spec = controllable_latent_spec
        self.parent_names = parent_names
        self.f_parent = f_parent
        self.f_child = f_child

        # used for controllable latent slot-based tracking. See `build` below
        self.controllable_latent_slot_index = 0
        self.num_children = num_children
        self._controllable_parent_slots = {name: None for name in self.parent_names}

    def build(self, nodes: Mapping[Text, Node]):
        for name, parent in nodes.items():
            if isinstance(parent, InfoNode) and True in tf.nest.flatten(parent.controllable_latent_spec):
                self._controllable_parent_slots[name] = parent.controllable_latent_slot_index
                parent.controllable_latent_slot_index += 1
                assert parent.controllable_latent_slot_index <= parent.num_children, \
                    'More children are registering slots to control this parent InfoNode than expected.'

    def train(self, experience: ts.NestedTensor) -> None:
        """Training for all `InfoNode`s.

        Args:
            experience: nested tensor of all `states` batched and for two units of time.
        """
        raise NotImplementedError('subclasses should define their own training method')

    @property
    def controllable_latent_spec(self):
        return self._controllable_latent_spec


def _f_parent_sample(states: ts.NestedTensor, parent_names: List[Text]):
    # select parent to pool data from by energy weighted bottom-up attention
    parent_energies = [states[name][keys.ENERGY] for name in parent_names]
    parent_dist = WeightedEmperical(logits=parent_energies, values=parent_names)
    parent_name = parent_dist.sample()
    parent_energy = states[parent_name][keys.ENERGY]
    parent_latent = states[parent_name][keys.LATENT]
    return parent_energy, parent_latent

def _f_parent_concat(states: ts.NestedTensor, parent_names: List[Text]):
    energies = ([states[name][keys.ENERGY] for name in parent_names])
    mean_energy = sum(energies) / len(energies)
    concat_latent = tf.nest.map_structure(lambda tensors: tf.concat(tensors, axis=1),
                                          [states[name][keys.LATENT] for name in parent_names])
    return mean_energy, concat_latent

def _f_parent_fixed_index_factory(index: int):
    def _f_parent_fixed_index(states: ts.NestedTensor, parent_names: List[Text]):
        parent_state = states[parent_names[index]]
        return parent_state[keys.ENERGY], parent_state[keys.LATENT]

def _f_child_sample(targets: List[Tuple[tf.Tensor, ts.NestedTensor]]):
    # select bias for top_down attention
    child_bias_dist = WeightedEmperical(
        logits=[-energy for energy, latent in targets],
        values=[(energy, latent[keys.ATTENTION_BIAS])
                for energy, latent in states[self.name][keys.TARGET_LATENTS]])
    return child_bias_dist.sample()
