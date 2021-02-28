from typing import List, Text, Callable, Optional

from .. import Node, InfoNode
from ...utils import types as ts


class AttnNode(InfoNode):

    def __init__(self,
                 observed_node_name: Text,
                 latent_spec: ts.NestedTensorSpec,
                 f_parent: Optional[Callable] = None,  # [[ts.NestedTensor, List[Text]], Tuple[tf.Tensor, ts.NestedTensor]]
                 f_child: Optional[Callable] = None,  # [[List[Tuple[tf.Tensor, ts.NestedTensor]]], Tuple[tf.Tensor, ts.NestedTensor]]
                 subnodes: Optional[List[Node]] = None,
                 name: Optional[str] = 'AttnNode'):
        """Meant to be called by subclass constructors.

        Args:
            observed_node_name: name of `Node` that will be attended to. If the observed node will be
                biased during top-down, it must be an `InfoNode`. The subclass may create its own observed
                node and store it in `subnodes`.
            latent_spec: TensorSpec nest. The structure of this `InfoNode`'s latent which may be observed
                by other `InfoNode`'s. The `location_spec` is also used to initialize this `InfoNode`'s
                `states[self.name][keys.STATES.LATENT]` with `tf.zeros` in matching shape.
            f_parent: Function that collects/selects parent information for this `InfoNode` on `bottom_up`.
            f_child: Function that collects/selects child information to utilize during `top_down`. If `None`,
                defaults to functions constructed from `info_node.functions.f_child_sample_factory`.
            subnodes: `Node` objects that are owned by this node.
            name: node name to attempt to use for variable scoping.
        """

        super(AttnNode, self).__init__(
            state_spec_extras=dict(),
            parent_names=[],
            latent_spec=latent_spec,
            f_parent=f_parent,
            f_child=f_child,
            subnodes=subnodes,
            name=name)

        self.observed_node_name = observed_node_name

    def train(self, experience: ts.NestedTensor) -> None:
        pass