from typing import Mapping, List, Text, Optional

import tf_agents as tfa

from ..utils import types as ts
from ..utils import keys
from . import InfoNode
from .info_node import functions


class RLNode(InfoNode):
    """Reinforcement learning `InfoNode.

    To interact with the `InfoNodeAgent`, `rl_agent.policy` must output a
    dictionary with the nodes it would like to control as keys.

    """


    def __init__(self,
                 rl_agent: tfa.agents.TFAgent,
                 input_node_names: List[Text],
                 output_node_names: List[Text],
                 recurrent_state_spec: ts.NestedTensorSpec,
                 name: Optional[Text] = 'RLNode'):

        super(RLNode, self).__init__(
                 state_spec_extras=dict(),
                 controllable_latent_spec=[],
                 parent_names=list(set(input_node_names) & set(output_node_names)),
                 num_children=0,
                 latent_spec=recurrent_state_spec,
                 f_parent=functions.f_parent_dict,
                 f_child=lambda targets: None,
                 subnodes=[],
                 name=name)

        self.rl_agent = rl_agent
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        time_step = ts.TimeStep(step_type=ts.StepType.MID, reward=0, discount=0.,
                                observation=self.f_parent(states=states, parent_names=self.input_node_names))
        policy_step: ts.PolicyStep = self.rl_agent.policy.action(time_step=time_step,
                                                                 policy_state=states[keys.STATES.LATENT])
        # save recurrent state
        states[self.name][keys.STATES.LATENT] = policy_step.state

        # assign output_node_names their respective values form the policy
        for output_node_name in self.output_node_names:
            slot_index = self._controllable_parent_slots[output_node_name]
            states[output_node_name][keys.STATES.TARGET_LATENTS][slot_index] \
                = (0., policy_step.action[output_node_name])

        return states

    def train(self, experience: ts.NestedTensor) -> None:
        policy_experience = tfa.utils.nest_utils.prune_extra_keys(self.rl_agent.training_data_spec, experience)
        self.rl_agent.train(experience=policy_experience)
