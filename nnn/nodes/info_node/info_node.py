from typing import Optional, Text, List, Mapping, Callable, Tuple, Union

import tensorflow as tf

from ...utils import types as ts
from ...utils import keys
from .. import Node
from . import functions


class InfoNode(Node):
    """
    The `InfoNode` corresponds to a node in a directed Bayesian graph. While all `Node` objects
    can recurrently alter the `states` during `bottom_up` and `top_down`, `InfoNode` objects also
    expose learning functions `train` and `loss`.

    While all `Node` objects may modify arbitrary properties of themselves and other info_node_names,
    the `InfoNode` also maintains three specific properties:
    * `LATENT`: `NestedTensor` containing its Bayesian-modeled latent variable.
    * `TARGET_LATENTS`: A list of 2-tuples of a 0-dimensional `Tensor` and a `NestedTensor`s identifying
            probabilistic energy and value of latent set points.
    * `ENERGY`: Probabilistic energy of latent (0-dimensional `Tensor`). Also representative of
            intrinsic loss. Used for weighting training replay.
    """

    def __init__(self,
                 state_spec_extras: Mapping[Text, ts.NestedTensorSpec],
                 parent_names: List[Text],
                 latent_spec: ts.NestedTensorSpec,
                 f_parent: Optional[Callable] = None,  # [[ts.NestedTensor, List[Text]], Tuple[tf.Tensor, ts.NestedTensor]]
                 f_child: Optional[Callable] = None,  # [[List[Tuple[tf.Tensor, ts.NestedTensor]]], Tuple[tf.Tensor, ts.NestedTensor]]
                 subnodes: Optional[List[Node]] = None,
                 name: Optional[str] = 'InfoNode'):
        """Meant to be called by subclass constructors.

        Args:
            state_spec_extras: The dict of (potentially further nested) variables to associate
                with this `InfoNode` during training/inference. Does not include `keys.STATES.ENERGY`,
                `LATENT`, or `TARGET_LATENTS`.
            parent_names: Names of parent `Node`'s, if any, that this `InfoNode` reads latent states
                from and possibly also biases by setting their `TARGET_LATENT` state. If `None`, defaults
                to `nnn.functions.f_parent_no_parents`.

                Since the graph must be static to be optimized, please invoke the `build` method with
                all this `InfoNode`'s parents' python objects. This gives each `InfoNode` an opportunity
                to allocate its particular parent-`TARGET_LATENT` interaction slot to prevent data collisions.
            latent_spec: TensorSpec nest. The structure of this `InfoNode`'s latent which may be observed
                by other `InfoNode`'s. The `location_spec` is also used to initialize this `InfoNode`'s
                `states[self.name][keys.STATES.LATENT]` with `tf.zeros` in matching shape.
            f_parent: Function that collects/selects parent information for this `InfoNode` on `bottom_up`.
            f_child: Function that collects/selects child information to utilize during `top_down`. If `None`,
                defaults to functions constructed from `info_node.functions.f_child_sample_factory`.
            subnodes: `Node` objects that are owned by this node.
            name: node name to attempt to use for variable scoping.
        """

        # TODO: `num_children` should not be a necesary variable.
        #       It should be possible to get this property after
        #       `build` is called.


        scalar_spec = tf.TensorSpec((1,))
        state_spec_dict = {
            keys.STATES.ENERGY: scalar_spec,
            keys.STATES.LATENT: latent_spec,
            keys.STATES.TARGET_LATENTS: num_children * [(
                scalar_spec, latent_spec
            )]
        }
        state_spec_dict.update(state_spec_extras)

        if f_parent is None:
            f_parent = functions.f_parent_no_parents

        if f_child is None:
            f_child = functions.f_child_sample_factory(0)

        if subnodes is None:
            subnodes = list()

        super(InfoNode, self).__init__(
            state_spec=state_spec_dict,
            name=name,
            subnodes=subnodes
        )
        self.parent_names = parent_names
        self.f_parent = f_parent
        self.f_child = f_child

        # used for controllable latent slot-based tracking. See `build` below
        self.controllable_latent_slot_index = 0
        self._controllable_parent_slots = {name: None for name in self.parent_names}
        self.nodes = dict()

    def build(self, nodes: List[Node]):
        """Initialize `InfoNode` parent target destination indeces.

        Args:
            nodes: a list containing all the nodes this `InfoNode` connects to.
        """
        nodes = list(set(self.nodes) | set(nodes))
        self.nodes = {node.name for node in nodes}

        for parent in nodes:
            if isinstance(parent, InfoNode) and True in tf.nest.flatten(parent.controllable_latent_spec):
                self._controllable_parent_slots[parent.name] = parent.controllable_latent_slot_index
                parent.controllable_latent_slot_index += 1
                assert parent.controllable_latent_slot_index <= parent.num_children, \
                    'More children are registering slots to control this parent InfoNode than expected.'

    def train(self, experience: ts.NestedTensor) -> None:
        """Training for all `InfoNode`s. This is not in `Node` because `Node`'s are not
        expected to have an energy property.

        Args:
            experience: nested tensor of all `states` batched and for two units of time.
        """
        raise NotImplementedError('subclasses should define their own training method')

    @property
    def controllable_latent_spec(self):
        return self._controllable_latent_spec

