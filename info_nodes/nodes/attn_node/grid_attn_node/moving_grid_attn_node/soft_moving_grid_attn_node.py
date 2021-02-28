import math
from typing import Text, List, Mapping, Optional, Union

import tensorflow as tf

from .moving_grid_attn_node import MovingGridAttnNode
from .. import InfoNode, TargetResponsiveNode
from ...utils import types as ts
from ...utils import keys


class SoftMovingGridAttnNode(MovingGridAttnNode):
    """Differentiable form of `MovingGridAttnNode` for small to medium sized workspaces"""

    def __init__(self,
                 observed_node_name: Text,
                 grid_shape: ts.Shape,
                 grid_space_manipulator_dimensions: Optional[List[ts.Int]] = None,
                 name: Optional[Text] = 'SoftMovingGridAttnNode'):

        self.grid_shape = grid_shape
        self.loc_node = LocNode(grid_shape=grid_shape)
        subnodes = [self.loc_node]

        if grid_space_manipulator_dimensions is None:
            grid_space_manipulator_dimensions = []
        for grid_space_manipulator_dimension in grid_space_manipulator_dimensions:
            subnodes.append(GridSpaceManipulatorNode(
                grid_hard_attn_node=self,
                manipulation_dimensions=grid_space_manipulator_dimension))

        super(SoftMovingGridAttnNode, self).__init__(
            observed_node_name=observed_node_name,
            latent_spec=grid_shape,
            subnodes = subnodes,
            name=name)

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        # TODO update latent with location
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        # TODO reach into `observednode` directly and modiefy its latent
        return states


class GridSpaceManipulatorNode(TargetResponsiveNode):
    """Manipulates the discreet rectangular (grid) space underlying
    a `SoftMovingGridAttnNode` by moving all the elements in a line, plane,
    cube, etc. of a the grid forward or backward.

    This Node is a `TargetResponsiveNode` so its latent is entirely
    indicitive of the last action taken.

    The latent is structured [2*D, 3] where D is the number of
    dimensions and 3 indicates the softmax of options for that axis:
      0. PUSH. insert a 0 and push all contents forward 1 grid space
      1. STILL. do nothing
      2. PULL. remove element at current location and pull back all contents
          back 1 grid space
    There are 2*D axes because each dimensional axis identifies two
    directions to manipulate the grid. They are ordered negative, then
    positive. In the case of multiples choices of PULL, conflicts are
    simply overiden in the order of the axis listing.

    For example:
    >>> GRID = [[1,  2,  3,  4], \
                [5,  6,  7,  8], \
                [9, 10, 11, 12]]
    >>> LOC = [1, 2]   # at the 7
    >>> latent = [[0.1, 0.8, -0.2], [0.3, 0.1, 0.05], \
                  [0.9, 0.2, 0.4], [0.2, 0.2, 0.7]]
    >>> # do 1D manipulation of GRID with latent at LOC. This executes:
    >>> #   1) STILL x-
    >>> #   2) PUSH x+
    >>> #   3) PUSH y-
    >>> #   4) PULL y+
    >>> # in order
    >>> GRID
    ... [[1,  2,  0,  4], \
         [5,  6, 11,  7], \
         [9, 10,  0, 12]]
    >>> LOC
    ... [1, 2] # unchanged; now at the 11
    """

    def __init__(self,
                 grid_hard_attn_node: SoftMovingGridAttnNode,
                 manipulation_dimensions: Union[1,2,3] = 1):
        super(GridSpaceManipulatorNode, self).__init__(
            latent_spec=ts.TensorSpec([2*grid_hard_attn_node.grid_shape.shape[0], 3]),
            name='GridSpaceManipulatorNode')

        self.grid_hard_attn_node = grid_hard_attn_node
        self.manipulation_dimensions = manipulation_dimensions

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        states = super(GridSpaceManipulatorNode).top_down(states)

        grid = states[self.grid_hard_attn_node.observed_node_name][keys.STATES.LATENT]
        loc = self.grid_hard_attn_node.loc_node.get_real_loc(states)
        batch_axes_op = tf.nn.softmax(states[self.name][keys.STATES.LATENT], axis=-1)

        grid_elems = tf.unstack(grid, axis=0)
        for batch_num, axes_op in enumerate(tf.unstack(batch_axes_op, axis=0)):
            grid_elem = grid_elems[batch_num]

            for half_axis_num, op in enumerate(tf.unstack(axes_op, axis=0)):
                axis_num = half_axis_num // 2
                direction = half_axis_num % 2

                # TODO: write this functions
                #   it should be fully differentiable.
                #   it will use binary forier transforms to do this.
                mask = get_mask(loc=loc, shape=grid.shape, axis=axis_num, direction=direction,
                                manipulation_dimensions=self.manipulation_dimensions)
                static_grid_elem = grid_elem * (1-mask)

                zeros_manifold = tf.zeros_like #  TODO
                zeros_at_start = tf.concat

                #### TODO ####
                # Depending on the direction and push/pull operation
                # concat zeros to the start/end of the grid tensor

                pushed_grid_elem = static_grid_elem + mask * tf.concat()
                still_grid_elem = grid_elem
                pulled_grid_elem = static_grid_elem + mask * tf.concat()
                grid_elem = op[0] * pushed_grid_elem \
                          + op[1] * still_grid_elem \
                          + op[2] * pulled_grid_elem

            grid_elems[batch_num] = grid_elem

        grid = tf.stack(grid_elems, axis=0)

        # perform action immediately because f_child_sample may not select it.
        states[self.grid_hard_attn_node.observed_node_name][keys.STATES.LATENT] = grid

        return states

    @staticmethod
    def get_mask(loc, shape, axis, direction, maipulation_dimensions):
        """Differentiably generates a mask of the grid space.
        Depending on the valued of `manipulation_dimensions`,
        elements that are in a line (1), plane (2), or cube (3)
        are close to 1. while remaining values are close to 0.

        Args:
            loc: The location to origonate the mask from.
            shape: The size of the grid space dimensions.
            axis: Axis to move along: 0, 1, 2, ...
            direction: 0=backward, 1=forward
            maipulation_dimensions: moving manifold dimensionality
                1=make a line starting at `loc` (a ray in geometry)
                2=make a plane
                3=cube
                ...

        Returns:
            a mask shaped: `shape` with 0=not moving, 1=moving
        """
        raise NotImplementedError()