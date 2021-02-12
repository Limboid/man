import math
import statistics
from typing import Optional, List, Text, Mapping

import tensorflow as tf

from ..utils import types as ts
from ..utils import keys, nest
from .info_node import InfoNode
from .info_node import functions

keras = tf.keras
unpacked_ave = statistics.mean  # special name just to make clear when I am wanting the average of an unpacked nest


class PredNode(InfoNode):
    """
    Predictive `InfoNode`

    f_abs, f_pred, and f_act must all output a `convert_to_tensor` distribution that can
    be sampled and entropy be calculated. All three functions are trained with Keras'
    `.fit` function using `negloglik`.
    """

    def __init__(self,
        f_abs: keras.Model,
        f_pred: keras.Model,
        f_act: keras.Model,
        parent_names: List[Text],
        neighbor_transl: Mapping[Text, keras.Model],
        num_children: ts.Int,
        latent_spec: ts.NestedTensorSpec,
        use_predictive_coding: bool = True,  # don't change this
        name: Text = 'PredNode'):

        self.f_abs = f_abs
        self.f_pred = f_pred
        self.f_act = f_act
        self.neighbor_transl = neighbor_transl
        self._use_predictive_coding = use_predictive_coding

        super(PredNode, self).__init__(
            state_spec_extras={
                keys.STATES.PRED_LATENT: latent_spec,
                keys.STATES.PRED_ENERGY: tf.TensorSpec((1,))
            },
            controllable_latent_spec=latent_spec,
            parent_names=parent_names,
            num_children=num_children,
            latent_spec=latent_spec,
            f_parent=functions.f_parent_dict,
            f_child=functions.f_child_sample_factory(beta=0.),
            subnodes=list(),
            name=name
        )

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:

        parent_energy, parent_latent_dict = self.f_parent(states=states, parent_names=self.parent_names)
        old_self_latent = states[self.name][keys.LATENT]
        energy = 0.

        # get new latent
        latent_dist = self.f_abs(dict(latent=old_self_latent, parent_latents=parent_latent_dict))
        latent_sample = latent_dist.sample()

        if self._use_predictive_coding:
            latent_sample = tf.nest.map_structure((lambda x, y: x-y),
                                                  states[self.name][keys.STATES.PRED_LATENT],
                                                  latent_sample)
        new_self_latent = latent_sample

        # update energy
        # psuedo-KL divergence loss
        kld_energy = nest.difference(old_self_latent, new_self_latent)
        energy = latent_dist.entropy() + -latent_dist.log_prob(latent_sample) + parent_energy + kld_energy
        states[self.name][keys.STATES.ENERGY] = energy

        # update self state
        states[self.name][keys.STATES.LATENT] = new_self_latent

        # return updated states
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:

        self_latent = states[self.name][keys.STATES.LATENT]
        energy = states[self.name][keys.STATES.ENERGY]

        # cluster to neighbors (echo chain style)
        neighbor_latent_dists = [tf.nest.flatten(f_transl(dict(latent=states[name][keys.STATES.LATENT],
                                                               energy=states[name][keys.STATES.ENERGY])).sample())
                                 for name, f_transl in self.neighbor_transl.items()]
        neighbor_latents = [dist.sample() for dist in neighbor_latent_dists]
        neighbor_energies = [states[name][keys.STATES.ENERGY] + dist.entropy() + -dist.log_prob(latent) \
                             - nest.difference(self_latent, latent) for latent, dist, name
                             in zip(neighbor_latents, neighbor_latent_dists, self.neighbor_transl.keys())]
        neighbor_energies.append(energy); neighbor_latents.append(self_latent)  # this allows self attention as well
        neighbor_weights = tf.nn.softmax(tf.constant(neighbor_energies))
        attended_latent = sum([tf.nest.map_structure(lambda x, y: x*y, weight, latent)
                               for weight, latent in zip(neighbor_weights, neighbor_latents)])
        self_latent = attended_latent

        # self prediction
        predicted_latent_dist = self.f_pred(dict(latent=self_latent, energy=energy))
        predicted_latent = predicted_latent_dist.sample()
        energy = energy + predicted_latent_dist.entropy() + -predicted_latent_dist.log_prob(predicted_latent)

        # target sampling
        targets = states[self.name][keys.STATES.TARGET_LATENTS]
        targets.append((energy, predicted_latent))
        target_energy, target_latent = self.f_child(targets=targets)

        # action generation
        action_dist = self.f_act(dict(
            latent=self_latent,
            target_latent=target_latent,
            latent_energy=energy,
            target_entropy=target_energy,
            parent_latents=self.f_parent(states=states, parent_names=self.parent_names)
        ))
        action_sample = action_dist.sample()
        energy = energy + action_dist.entropy() + -action_dist.log_prob(action_sample)

        # assign parent targets
        for parent_name in self.parent_names:
            if parent_name in action_sample:
                slot_index = self._controllable_parent_slots[parent_name]
                states[parent_name][keys.STATES.TARGET_LATENTS][slot_index] = (energy, action_sample[parent_name])

        # update states
        states[self.name][keys.STATES.PRED_LATENT] = predicted_latent
        states[self.name][keys.STATES.ENERGY] = energy

        return states

    def train(self, experience: ts.NestedTensor) -> None:
        pass