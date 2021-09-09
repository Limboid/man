"""
high level NNN interface

maybe not framework agnostic
"""

from typing import Mapping, List, Iterable, Union, Callable, NoReturn

from nnn.utils.nest import map_nested
from nnn.utils.types import NNNState, NodeState
from nnn.nnn import NNN
from nnn.nodes import Node


class Agent:

    def __init__(self,
                 nnn: NNN,
                 inputs: Union[int, List[str], List[Node]],
                 outputs: Union[int, List[str], List[Node]],
                 hparams: Mapping = None):

        # make list representation for input and output labels

        raise NotImplementedError()
        self.built = False


    def __call__(self,
                 inputs,
                 state=None,
                 *nnn_args,
                 **nnn_kwargs):

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

            # build temporary hparams structure
            tmp_hparams = self.hparams.copy()
            if hparam_overrides is not None and len(hparam_overrides) > 0:
                tmp_hparams.update(hparam_overrides)

            # perform internal updates
            for cycle in range(tmp_hparams['internal_updates']):
                # update internal state
                state = self._update_node_network_state(state)

                # run all internal update callbacks
                for cb in tmp_hparams['internal_update_callbacks']:
                    cb(state)

            return state

        if not self.built:
            self.build(map_nested(

            ))

        if state is None:
            state = self.state

        # assign input node values
        # convert to tensor if not already


        # return output node values
        raise NotImplementedError()

    def train(self, state_trajectory: Iterable[NNNState]):
        raise NotImplementedError()

    def fit(self, dataset):
        raise NotImplementedError()