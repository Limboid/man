from typing import Optional, Mapping, List, Tuple, Union, Text, Callable

import tensorflow as tf

from ...utils import types as ts
from ...utils import keys
from ..node import Node
from .. import info_node
from ..info_node import InfoNode


class SymbolWorkspaceNode(Node):
    """Container class for symbolic processing. `SymbolWorkspaceNode` creates subnodes on initialization which do
    most of the real work. It is useful for giving the agent a mental workspace to manipulate information. This
    node stores data in the following keys in `states`:

    - `states[self.name][keys.STATES.ENERGY]`: standrard probabilistic-loss-energy scalar. It is set constant (0) in
        the base implementation but could be used as a regularizer.

    - `states[self.name][keys.STATES.WORKSPACE]`: all locations (both attented and hidden) in the symbol space
        &emdash; at least, the symbolc space the is stored in the tensorflow graph. Dynamic database or extended text
        processing capability could be implemented in custom subclasses where the most seemingly relevant items are
        retrieved into the graph memory on `bottom_up` and `top_down`.

    The actual reading and writing functionality is managed by `(SymbolLocNode, SymbolProcessingNode)` pairs of
    subnodes. Separating read/write functionality into separate nodes avoids complicating the latent structure with
    nests of `'location'` and `'true_latent'`. The `SymbolLocNode` updates the location for its associated
    `SymbolProcessingNode` which in turn reads or writes data over a subset of the container class' `WORKSPACE`.
    See the aforementioned classes for details.

    This base class and its base `SymbolLocNode` and `SymbolProcessingNode` child classes are totally functional-
    programming-oriented. However they can also be subclassed for finer control."""

    def __init__(self,
                 get_window_fn: Callable,
                 put_window_fn: Callable,
                 location_spec: ts.NestedTensorSpec,
                 delta_loc_spec: ts.NestedTensorSpec,
                 workspace_spec: ts.NestedTensorSpec,
                 rw_fns_and_specs: List[Tuple[Union[Callable,None], Union[Callable,None]],
                                        Union[ts.NestedTensorSpec, None], Union[ts.NestedTensorSpec, None]],
                 change_loc_fn: Callable,
                 name: Optional[Text] = 'SymbolWorkspaceNode'):
        """Creates `SymbolWorkspaceNode` and associated subnodes.

        NOTE: This initializer assumes all symbol-processing functions operate
        on homeogenous windows. If this is not your case, consider creating a
        `SymbolWorkspaceNode` without heterogenous window functions and then
        manually add `LambdaNode`'s to perform the remaining data processing. For
        finer-grained control, you might subclass `SymbolWorkspaceNode`.

        Args:
            get_window_fn: Gets a window from the workspace at a specified location.
                Informally, `(workspace: Tws, location: Tloc) -> Twindow`.
            put_window_fn: Overrides data in the current window to the workspace at a
                specified location. Informally, `(workspace: Tws, location: Tloc, new_window: Twindow) -> Tws`.
            location_spec: `ts.NestedTensorSpec` specifying the structure of the location
                tensor nest used by `get_window_fn` and `put_window_fn`.
            delta_loc_spec: TensorSpec nest. Used by `change_loc_fn` to change the location in symbol space.
            workspace_spec: `ts.NestedTensorSpec` specifying the structure of the workspace
                tensor nest used by `get_window_fn` and `put_window_fn`.
            rw_fns_and_specs: List of quadtruple-paired read and write functions and their associated output
                latent and input target `TensorSpec`'s respectively. Specifically, at each symbol processing
                head. Specifically, they are given as a tuple
                `(read_fn, write_fn, location_spec, controllable_latent_spec)` where:
                - `read_fn(window: Twindow) -> ts.NestedTensor` outputs a new latent on `bottom_up`.
                    If `None`, it is `location_spec` is also ignored and no values are read on `bottom_up`.
                - `write_fn(window: Twindow, target: ts.NestedTensor) -> Twindow` modifies the window on
                    `top_down`. If `None`, target latents are ignored and other nodes cannot controll this
                    node on `top_down`.
                - `location_spec` defines the tensor structure of the output of `read_fn` if not `None`.
                - `controllable_latent_spec` defines the tensor structure of `Tdloc` used to update the
                    location on `top_down`. `(See `change_loc_fn`.)
            change_loc_fn: This function updates the location in symbol space. It also ensures the
                location remains valid. Its signature is `(old_loc: Tloc, delta_loc: Tdloc) -> Tloc`.
            name: Optional name for the `SymbolWorkspaceNode`."""

        state_spec = {keys.STATES.ENERGY: ts.scalar_tensor_spec,
                      keys.STATES.WORKSPACE: workspace_spec}

        subnodes = []
        for read_fn, write_fn, latent_spec, controllable_latent_spec in rw_fns_and_specs:
            symbol_loc_node = SymbolLocNode(
                location_spec=location_spec,
                delta_loc_spec=delta_loc_spec,
                change_loc_fn=change_loc_fn,
                name=f'{name}_SymbolLocNode')
            symbol_processing_node = SymbolProcessingNode(
                read_fn=read_fn,
                write_fn=write_fn,
                get_window_fn=get_window_fn,
                put_window_fn=put_window_fn,
                container_node_name=name,
                symbol_loc_node_name=symbol_loc_node.name,
                latent_spec=latent_spec,
                controllable_latent_spec=controllable_latent_spec,
                name=f'{name}_SymbolProcessingNode')
            subnodes.append(symbol_loc_node)
            subnodes.append(symbol_processing_node)

        super(SymbolWorkspaceNode, self).__init__(
            state_spec=state_spec,
            subnodes=subnodes,
            name=name)


class SymbolLocNode(InfoNode):
    """Child class for symbolic processing. This class manages the location for an associated `SymbolProcessingNode`."""

    def __init__(self,
                 location_spec: ts.NestedTensorSpec,
                 delta_loc_spec: ts.NestedTensorSpec,
                 change_loc_fn: Callable,
                 name: Text = 'SymbolLocNode'):
        """Meant to be called by container class `SymbolWorkspaceNode`.

        Args:
            location_spec: TensorSpec nest. The structure of this `SymbolLocNode`'s latent which may be observed
                by other `InfoNode`'s represents its location in symbol space. The `latent_spec` is also used to
                initialize this `SymbolLocNode`'s `states[self.name][keys.STATES.LATENT]` with `tf.zeros` in
                matching shape.
            delta_loc_spec: TensorSpec nest. Used by `change_loc_fn` to change the location in symbol space.
            change_loc_fn: This function updates the location in symbol space. It also ensures the location
                remains valid. Its signature is `(old_loc: Tloc, delta_loc: ts.NestedTensor) -> Tloc`.
            name: Name of this `SymbolProcessingNode`.
        """
        self.change_loc_fn = change_loc_fn

        super(SymbolLocNode, self).__init__(
            state_spec_extras=dict(),
            controllable_latent_spec=delta_loc_spec,
            parent_names=[],
            latent_spec={keys.STATES.SYMBOLIC.LOC: location_spec,
                         keys.STATES.SYMBOLIC.DELTA: delta_loc_spec},
            f_parent=info_node.functions.f_parent_no_parents,
            f_child=info_node.functions.f_child_sample_factory(beta=0.),
            name=name)

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        old_loc = states[self.name][keys.STATES.LATENT]
        child_sample_energy, delta_loc = self.f_child(states[self.name][keys.STATES.TARGET_LATENTS])

        change_loc_energy, new_loc = self.change_loc_fn(old_loc=old_loc, delta_loc=delta_loc)

        states[self.name][keys.STATES.ENERGY] = child_sample_energy + change_loc_energy
        states[self.name][keys.STATES.LATENT] = {keys.STATES.SYMBOLIC.LOC: new_loc,
                                                 keys.STATES.SYMBOLIC.DELTA: delta_loc,}

        return states


class SymbolProcessingNode(InfoNode):
    """Child class for symbolic processing. This node reads and/or writes to a subset of its container class' global
    workspace determined by initialzer-passed parameters `get_window_fn`, `put_window_fn`, and the location value of
    its associated `SymbolLocNode`."""

    def __init__(self,
                 read_fn: Callable,
                 write_fn: Callable,
                 get_window_fn: Union[Callable,None],
                 put_window_fn: Union[Callable,None],
                 container_node_name: Text,
                 symbol_loc_node_name: Text,
                 latent_spec: ts.NestedTensorSpec,
                 controllable_latent_spec: ts.NestedTensorSpec,
                 name: Text = 'SymbolProcessingNode'):
        """Meant to be called by container class `SymbolWorkspaceNode`.

        Args:
            read_fn: (window: Twindow) -> ts.NestedTensor` outputs a new latent on `bottom_up`.
                If `None`, no latent is updated on `bottom_up` and the latent spec is assigned
                a constant scalar value.
            write_fn: `(window: Twindow, target: ts.NestedTensor) -> Twindow'`
                modifies the window on `top_down`. If `None`, the window is not updated on
                `top_down` and `num_children` is assigned to 0.
            get_window_fn: Gets a window from the workspace at a specified location.
                Informally, `(workspace: Tws, location: Tloc) -> Twindow`.
            put_window_fn: Overrides data in the current window to the workspace at a
                specified location. Informally, `(workspace: Tws, location: Tloc, new_window: Twindow) -> Tws`.
            container_node_name: Name of `SymbolWorkspaceNode` to get/put data from/to.
            symbol_loc_node_name: Name of paired `SymbolLocNode` to get location from.
            latent_spec: TensorSpec nest. The structure of this `InfoNode`'s latent which may be observed
                by other `InfoNode`'s. The `location_spec` is also used to initialize this `InfoNode`'s
                `states[self.name][keys.STATES.LATENT]` with `tf.zeros` in matching shape.
            controllable_latent_spec: TensorSpec nest. The subset of this `InfoNode`'s latent that is
                controllable by its children.
            name: Name of this `SymbolProcessingNode`.
        """
        self.read_fn = read_fn
        self.write_fn = write_fn
        self.get_window_fn = get_window_fn
        self.put_window_fn = put_window_fn
        self.container_node_name = container_node_name
        self.symbol_loc_node_name = symbol_loc_node_name

        if self.read_fn is None:
            self.bottom_up = lambda states: states
        if self.write_fn is None:
            assert self.num_children == 0, f'This `SymbolProcessingNode` {self.name}' \
                                            'cannot have children without a `write_fn`.'
            self.top_down = lambda states: states

        super(SymbolProcessingNode, self).__init__(
            state_spec_extras=dict(),
            controllable_latent_spec=controllable_latent_spec,
            parent_names=[],
            latent_spec=latent_spec,
            f_parent=info_node.functions.f_parent_no_parents,
            f_child=info_node.functions.f_child_sample_factory(beta=0.),
            name=name,
        )

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        workspace = states[self.container_node_name][keys.STATES.WORKSPACE]
        location = states[self.symbol_loc_node_name][keys.STATES.LATENT][keys.STATES.SYMBOLIC.LOC]
        window = self.get_window_fn(workspace=workspace, location=location)

        states[self.name][keys.STATES.LATENT] = self.read_fn(window=window)

        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        workspace = states[self.container_node_name][keys.STATES.WORKSPACE]
        location = states[self.symbol_loc_node_name][keys.STATES.LATENT][keys.STATES.SYMBOLIC.LOC]
        window = self.get_window_fn(workspace=workspace, location=location)

        energy, target = self.f_child(targets=states[self.name][keys.STATES.TARGET_LATENTS])
        states[self.name][keys.STATES.ENERGY] = energy
        new_window = self.write_fn(window=window, target=target)
        states[self.container_node_name][keys.STATES.WORKSPACE] = self.put_window_fn(
            workspace=workspace, location=location, new_window=new_window)

        return states
