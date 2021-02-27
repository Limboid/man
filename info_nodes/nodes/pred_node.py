import logging
import statistics
from typing import Optional, List, Text, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

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
                 optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                 name: Text = 'PredNode'):

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)

        super(PredNode, self).__init__(
            state_spec_extras={
                keys.STATES.PRED_LATENT: latent_spec,
                keys.STATES.PRED_ENERGY: tf.TensorSpec((1,))
            },
            controllable_latent_spec=latent_spec,  # don't change this unless you know what you're doing.
            parent_names=parent_names,
            num_children=num_children,
            latent_spec=latent_spec,
            f_parent=functions.f_parent_dict,
            f_child=functions.f_child_sample_factory(beta=0.),
            subnodes=list(),
            name=name
        )

        self.f_abs = f_abs
        self.f_pred = f_pred
        self.f_act = f_act
        self.neighbor_transl = neighbor_transl
        self._use_predictive_coding = use_predictive_coding
        self.optimizer = optimizer

        self.sample_action = None

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:

        parent_energy, parent_latent_dict = self.f_parent(states=states, parent_names=self.parent_names)
        old_self_latent = states[self.name][keys.STATES.LATENT]

        # get new latent
        latent_dist = self.f_abs(dict(latent=old_self_latent, parent_latents=parent_latent_dict))
        latent_sample = latent_dist.sample()

        if self._use_predictive_coding:
            latent_sample = tf.nest.map_structure((lambda x, y: x - y),
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
                                                               dest_energy=states[self.name][keys.STATES.ENERGY],
                                                               src_energy=states[name][keys.STATES.ENERGY])).sample())
                                 for name, f_transl in self.neighbor_transl.items()]
        neighbor_latents = [dist.sample() for dist in neighbor_latent_dists]
        neighbor_energies = [states[name][keys.STATES.ENERGY] + dist.entropy() + -dist.log_prob(latent) \
                             - nest.difference(self_latent, latent) for latent, dist, name
                             in zip(neighbor_latents, neighbor_latent_dists, self.neighbor_transl.keys())]
        neighbor_energies.append(energy)
        neighbor_latents.append(self_latent)  # this allows self attention as well
        neighbor_weights = tf.nn.softmax(tf.constant(neighbor_energies))
        attended_latent = sum([tf.nest.map_structure(lambda x, y: x * y, weight, latent)
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
                states[parent_name][keys.STATES.TARGET_LATENTS][slot_index] \
                    = (energy, action_sample[parent_name])
        if not self.sample_action:
            self.sample_action = action_sample

        # update states
        states[self.name][keys.STATES.PRED_LATENT] = predicted_latent
        states[self.name][keys.STATES.ENERGY] = energy

        return states

    def train(self, experience: ts.NestedTensor) -> None:
        for i in range(3):
            logging.log(1, f'PredNode {self.name} begining training epoch {i}')
            self._train_epoch(experience=experience)

    def _train_epoch(self, experience: ts.NestedTensor) -> None:

        # TODO
        #  Double-check this code.
        #  I'm not sure if I kept in mind that
        #  the experience records what `states` looks like
        #  at the end on the round - rather than the beginning.

        def information_loss(ytrue, ypred):
            if isinstance(ytrue, tfp.distributions.Distribution) and isinstance(ypred, tfp.distributions.Distribution):
                loss = ypred.kl_divergence(ytrue)
            elif isinstance(ytrue, ts.NestedTensor) and isinstance(ypred, tfp.distributions.Distribution):
                loss = ypred.log_prob(ytrue)
            else:
                return tf.losses.mse(ytrue, ypred)
            return tf.reduce_sum(loss)

        trainable_vars = [f_trans.trainable_variables
                          for _, f_trans in self.neighbor_transl.items()] + \
                         [self.f_abs.trainable_variables,
                          self.f_pred.trainable_variables,
                          self.f_act.trainable_variables]

        with tf.GradientTape() as tape:
            # this code operates on all timesteps simultaneously so we are going to ro
            exp = experience
            exp_next = tf.nest.map_structure(lambda t: tf.roll(t, shift=1, axis=1), exp)
            exp_prev = tf.nest.map_structure(lambda t: tf.roll(t, shift=-1, axis=1), exp)

            exp_prev = tf.nest.map_structure(lambda t: t[:, 1:-1, ...], exp_prev)
            exp = tf.nest.map_structure(lambda t: t[:, 1:-1, ...], exp)
            exp_next = tf.nest.map_structure(lambda t: t[:, 1:-1, ...], exp_next)

            # exp_prev_flat = nest.flatten_time_into_batch_axis(self.bottom_up(exp_prev))
            exp_flat = nest.flatten_time_into_batch_axis(self.bottom_up(exp))
            exp_next_flat = nest.flatten_time_into_batch_axis(self.bottom_up(exp_next))

            exp_flat_scrambled = tf.nest.map_structure(lambda t: tf.roll(t, shift=t.shape[0]//2, axis=0), exp_flat)

            # train f_trans by association
            # use positive and negative sampling
            latent_trans = [f_trans(exp_flat[neighbor_name][keys.STATES.LATENT])
                            for neighbor_name, f_trans in self.neighbor_transl.items()]

            # train f_abs and f_pred to minimize predictive error
            exp_before_self_bottom_up = exp
            exp_before_self_bottom_up[self.name][keys.STATES.ENERGY] = exp_prev[self.name][keys.STATES.ENERGY]
            exp_before_self_bottom_up[self.name][keys.STATES.LATENT] = exp_prev[self.name][keys.STATES.LATENT]
            exp_flat_before_self_bottom_up = nest.flatten_time_into_batch_axis(exp_before_self_bottom_up)
            exp_flat_after_bottom_up = self.bottom_up(states=exp_flat_before_self_bottom_up)
            exp_flat_after_top_down = self.top_down(states=exp_flat_after_bottom_up)
            self_after_top_down = exp_flat_after_top_down[self.name]
            # now [keys.STATES.PRED_LATENT] and ENERGY are set

            # train f_act from inverse modeling trajectory data
            ideal_parent_targets = {name: self.nodes[name].structure_latent_as_controllable(
                                        exp_next_flat[name][keys.STATES.LATENT])
                                    for name in self.parent_names}
            actual_parent_targets = {name: exp_flat_after_top_down[name][keys.STATES.TARGET_LATENTS]
                                        [self.controllable_latent_slot_index[name]]
                                     for name in self.parent_names}

            error = nest.difference(exp_flat, latent_trans, information_loss) \
                    - nest.difference(exp_flat_scrambled, latent_trans, information_loss) \
                    + self_after_top_down[keys.STATES.ENERGY] \
                    + self_after_top_down[keys.STATES.PRED_ENERGY] \
                    + nest.difference(exp_next_flat[self.name][keys.STATES.LATENT],
                                      self_after_top_down[keys.STATES.PRED_LATENT],
                                      information_loss) \
                    + nest.difference(ideal_parent_targets,
                                      actual_parent_targets,
                                      information_loss)

        # optimize
        grads = tape.gradient(error, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
