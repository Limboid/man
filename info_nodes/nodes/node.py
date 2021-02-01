from __future__ import annotations

import tensorflow as tf

from typing import Optional, List, Text, Mapping
import tf_agents.typing.types as ts


class Node:
    """Node base class

    This class provides the following variables for use by other `Node` objects and the `InfoNodeAgent`:
    * state_spec: a nested structure specifying the tensor shapes of its state used during training/inference.
    * initial_state: an initial state to pass in during training/inference.
    """

    _UNIQUE_NAME_COUNTER = {}

    def __init__(
            self,
            state_spec: ts.NestedTensorSpec,
            name: Optional[Text] = 'Node',
            subnodes: Optional[List[Node]] = None):
        """Meant to be called by subclass constructors.

        Args:
            state_spec: the nested structure of variables to associate
                with this `Node` during training/inference.
            name: node name to attempt to use for variable scoping.
            subnodes: `Node` objects that are owned by this node.
        """
        self._state_spec = state_spec
        self._name = self._make_name_unique(name)
        self._subnodes = subnodes if subnodes is not None else list()

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        """Perception

        Generally used to observe information from parents and make appropriate internal
        state changes. However, nodes can arbitrarily alter data allowing organ nodes to
        competitively consume energy from energy-producing nodes.

        Args:
            states: states of all nodes in the InfoNodePolicy

        Returns:
            updated states dict containing all nodes in the InfoNodePolicy
        """
        pass

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        """Action

        Generally used to make demands to parent nodes. However, nodes can arbitrarily
        alter data allowing organ nodes to competitively consume energy from energy-producing
        nodes. Loss for this round should be assigned into the `states` slot for this node.

                Args:
                    states: states of all nodes in the InfoNodePolicy

                Returns:
                    new states for all nodes in the InfoNodePolicy
                """
        pass

    def initial_state(self, batch_size: Optional[ts.Int]) -> ts.NestedTensor:
        """

        Args:
            batch_size: Tensor or constant: size of the batch dimension. Can be None
                in which case no dimensions gets added.

        Returns:
            `Nested` structure of `Tensor`s to initialize this `Node`'s state with
                during training/inference
        """
        shape_fn = (lambda x: x)
        if batch_size is not None:
            shape_fn = (lambda x: (batch_size,) + x)

        return tf.nest.map_structure(shape_fn, self.state_spec)

    @property
    def state_spec(self):
        return self._state_spec

    @property
    def name(self):
        return self._name

    @property
    def subnodes(self):
        return self._subnodes

    @classmethod
    def _make_name_unique(cls, name):
        """Sequentially suffixes names. Non-idempotent method to ensure no name collisions.

        Example:
            >>> node = Node()
            >>> node._make_name_unique('Node')
            'Node1'
            >>> node._make_name_unique('Node')
            'Node2'
            >>> node._make_name_unique('Node')
            'Node3'
            >>> node._make_name_unique('Node1')
            'Node11'

        Args:
            name: Node instance name to make unique.

        Returns: unique name
        """
        if name in Node._UNIQUE_NAME_COUNTER:
            Node._UNIQUE_NAME_COUNTER[name] += 1
        else:
            Node._UNIQUE_NAME_COUNTER[name] = 1
        return name + str(Node._UNIQUE_NAME_COUNTER[name])
