from typing import List, Mapping, Any

from nnn.nodes import Node
from nnn.connections import Connection


NNNState = Mapping[Connection, Any]


class NNN():

    DEFAULT_HPARAMS = dict(
         internal_updates=1,
         internal_update_callbacks=[]
    )

    def __init__(self,
                 nodes: List[Node],
                 connections: List[Connection],
                 hparams: Mapping[str, Any] = None):

        self.hparams = NNN.DEFAULT_HPARAMS.copy()
        if hparams is not None and len(hparams) > 0:
            self.hparams.update(hparams)

        self.nodes = nodes
        self.connections = connections
        self.built = False

    def build(self):
        """Builds nodes"""
        for node in nodes:
            node.build()

    def get_initial_state(self, batch_size: int = 1) -> NNNState:
        raise NotImplementedError()

    def __call__(self, state: NNNState, **hparam_overrides) -> NNNState:
        """Updates `state`. This is the primary user-facing function of `NNN`.

        Args:
            state: A `NNNState` structure containing the internal state of
                the network after the previous update step.
            **hparam_overrides: A `dict` of overrides to temporarily apply
                to hparams (such as 'internal_updates') for this function call.

        Returns:
            Returns the updated `state`.
        """
        for node in self.nodes:
            incoming = [
                (conn, state[conn])
                for conn in self.connections
                if conn.dst_port == node.]
            outgoing = node(incoming)

