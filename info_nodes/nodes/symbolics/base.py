from typing import Optional, List, Tuple, Union, Text, Callable

import tensorflow as tf

from ...utils import types as ts
from ...utils import keys
from ..node import Node
from ..info_node import InfoNode


class SymbolicWorkspaceNode(Node):
    """Container class for symbolic processing. `SymbolicWorkspaceNode` creates subnodes on initialization which do
    most of the real work. It is useful for giving the agent a mental workspace to manipulate information. This
    node stores data in the following keys in `states`:

    - `states[self.name][keys.STATES.ENERGY]`: standrard probabilistic-loss-energy scalar. It is set constant (0) in
        the base implementation but could be used as a regularizer.

    - `states[self.name][keys.STATES.WORKSPACE]`: all locations (both attented and hidden) in the symbol space
        &emdash; at least, the symbolc space the is stored in the tensorflow graph. Dynamic database or extended text
        processing capability could be implemented in custom subclasses where the most seemingly relevant items are
        retrieved into the graph memory on `bottom_up` and `top_down`.

    The actual reading and writing functionality is managed by `(SymbolicLocNode, SymbolProcessingNode)` pairs of
    subnodes. Separating read/write functionality into separate nodes avoids complicating the latent structure with
    nests of `'location'` and `'true_latent'`. The `SymbolicLocNode` updates the location for its associated
    `SymbolProcessingNode` which in turn reads or writes data over a subset of the container class' `WORKSPACE`.
    See the aforementioned classes for details.

    This base class and its base `SymbolicLocNode` and `SymbolProcessingNode` child classes are totally functional-
    programming-oriented. However they can also be subclassed for finer control."""

    def __init__(self,
                 get_window_fn: Callable,
                 put_window_fn: Callable,
                 location_spec: ts.NestedTensorSpec,
                 workspace_spec: ts.NestedTensorSpec,
                 rw_fns: List[Tuple[Union[Callable,None], Union[Callable,None]]],
                 name: Optional[Text] = 'SymbolicWorkspaceNode'):
        """Creates `SymbolicWorkspaceNode` and associated subnodes.

        NOTE: This initializer assumes all symbol-processing functions operate
        on homeogenous windows. If this is not your case, consider creating a
        `SymbolicWorkspaceNode` without heterogenous window functions and then
        manually add `LambdaNode`'s to perform the remaining data processing. For
        finer-grained control, you might subclass `SymbolicWorkspaceNode`.

        Args:
            get_window_fn: Gets a window from the workspace at a specified location.
                Informally, `(workspace: Tws, location: Tloc) -> Twindow`.
            put_window_fn: Overrides data in the current window to the workspace at a
                specified location. Informally, `(window: Twindow, location: Tloc) -> Tws`.
            location_spec: `ts.NestedTensorSpec` specifying the structure of the location
                tensor nest used by `get_window_fn` and `put_window_fn`.
            workspace_spec: `ts.NestedTensorSpec` specifying the structure of the workspace
                tensor nest used by `get_window_fn` and `put_window_fn`.
            rw_fns: List of paired read and write functions at each symbol processing head.
                Specifically, they are given as a tuple `(read_fn, write_fn)` where
                `read_fn(window: Twindow) -> ts.NestedTensor` outputs a new latent on `bottom_up`
                and `write_fn(window: Twindow, target_latent: ts.NestedTensor) -> Twindow'`
                modifies the window on `top_down`. If `read_fn` or `write_fn` is None, it is
                not included.
            name: Optional name for the `SymbolicWorkspaceNode`."""

        state_spec = {
            keys.STATES.ENERGY: ts.ScalarTensorSpec,
            keys.STATES.WORKSPACE: workspace_spec}

        subnodes = [SymbolProcessingNode()
                    for read_fn, write_fn in rw_fns]

        super(SymbolicWorkspaceNode, self).__init__(
            state_spec=state_spec,
            subnodes=subnodes,
            name=name)


class SymbolicLocNode(InfoNode):
    """Child class for symbolic processing. This class manages the location for an associated `SymbolProcessingNode`."""

    pass


class SymbolProcessingNode(InfoNode):
    """Child class for symbolic processing. This node reads and/or writes to a subset of its container class' global
    workspace determined by initialzer-passed parameters `get_window_fn`, `put_window_fn`, and the location value of
    its associated `SymbolicLocNode`."""

    def __init__(self,
                 read_fn: Callable,
                 write_fn: Callable,
                 get_window_fn: Union[Callable,None],
                 put_window_fn: Union[Callable,None],
                 container_name: Text,
                 name):
        """Meant to be called by container class `SymbolicWorkspaceNode`.

        Args:
            read_fn: (window: Twindow) -> ts.NestedTensor` outputs a new latent on `bottom_up`.
                If `None`, no latent is updated on `bottom_up` and the latent spec is assigned
                a constant scalar value.
            write_fn: `(window: Twindow, target_latent: ts.NestedTensor) -> Twindow'`
                modifies the window on `top_down`. If `None`, the window is not updated on
                `top_down` and `num_children` is assigned to 0.
            get_window_fn: Gets a window from the workspace at a specified location.
                Informally, `(workspace: Tws, location: Tloc) -> Twindow`.
            put_window_fn: Overrides data in the current window to the workspace at a
                specified location. Informally, `(window: Twindow, location: Tloc) -> Tws`.
            container_name: name of `SymbolicWorkspaceNode` to get/put data from/to.
            name:
        """

        pass

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        pass

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        pass