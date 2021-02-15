from typing import Optional, List, Text, Callable

from ...utils import types as ts
from ...utils import keys
from ..node import Node
from ..info_node import InfoNode


class SymbolicNode(InfoNode):
    """Symbol processing node.
    Useful for giving the agent a mental workspace to manipulate information. This node
    stores data in `states[self.name][keys.STATES.GLOBAL_WORKSPACE]` and
    `states[self.name][keys.STATES.LOCATION]`.

    Its `data_read_fns` and `data_write_fns` are called on `bottom_up` and `top_down` respectively.
    Instead of operating on a global workspace (gws), they receive a 'window' of data from
    `read_window_fn`. Afterwards, data is written back to the global workspace by `write_window_fn`.
    The location used for taking windowed subsets of the global workspace is also updated by
    read and write functions `location_to_latent` and `update_location_latent` which update the latent
    state that is read by other nodes.

    These functions are called in the order listed below. Their signatures are:
    - `location_to_latent(target: ts.NestedTensor, location: Tloc) ->
    - `read_window_fn(working_data_spec: Tgws, location: Tloc) ->
    - `data_read_fn(data_window: Twindow) -> ts.NestedTensor`
    - `data_write_fn(data_window: Twindow, targets: ts.NestedTensor) -> Twindow`
    - `write_window_fn(data_window: Twindow, location: Tloc) -> Tgws
    - `target_to_location(target: ts.NestedTensor, location: Tloc) -> Tloc
    where `Tgws`, `Tloc`, and `Twindow` are particular `ts.NestedTensorSpec` types used to
    designate the global worspace, location, and window.
    """

    def __init__(self,
                 location_spec: ts.NestedTensorSpec,
                 working_data_spec: ts.NestedTensorSpec,
                 f_window: Callable,
                 read_fns: List[Callable],
                 write_fns: List[Callable],
                 name: Optional[List[Text]] = 'SymbolicNode'):

        super(SymbolicNode, self).__init__(
            state_spec_extras={
                keys.STATES.LOCATION: location_spec,
                keys.STATES.GLOBAL_WORKSPACE: working_data_spec,
            },
            subnodes=[reader_subnodes+writer_subnodes],
            name=name)
