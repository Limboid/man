from typing import Optional, Text, List, Mapping, Callable, Tuple, Union, Counter

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from ...utils import types as ts
from ...utils import keys


def f_parent_no_parents(states: ts.NestedTensor, parent_names: List[Text]):
    return


def f_parent_sample(states: ts.NestedTensor, parent_names: List[Text]):
    # select parent to pool data from by energy weighted bottom-up attention
    parent_energies = [states[name][keys.STATES.ENERGY] for name in parent_names]
    parent_dist = tfd.OneHotCategorical(logits=parent_energies)
    parent_name = parent_names[tf.argmax(parent_dist.sample())]
    parent_energy = states[parent_name][keys.STATES.ENERGY]
    parent_latent = states[parent_name][keys.STATES.LATENT]
    return parent_energy, parent_latent


def f_parent_concat(states: ts.NestedTensor, parent_names: List[Text]) -> Tuple[tf.Tensor, ts.NestedTensor]:
    energies = ([states[name][keys.STATES.ENERGY] for name in parent_names])
    mean_energy = sum(energies) / len(energies)
    concat_latent = tf.nest.map_structure(lambda tensors: tf.concat(tensors, axis=1),
                                          [states[name][keys.STATES.LATENT] for name in parent_names])
    return mean_energy, concat_latent


def f_parent_dict(states: ts.NestedTensor, parent_names: List[Text]) -> Tuple[tf.Tensor, ts.NestedTensor]:
    energies = [states[name][keys.STATES.ENERGY] for name in parent_names]
    mean_energy = sum(energies) / len(energies)
    latents_dict = {name: states[name][keys.STATES.LATENT] for name in parent_names}
    return mean_energy, latents_dict


def f_parent_fixed_index_factory(index: int):
    def _f_parent_fixed_index(states: ts.NestedTensor, parent_names: List[Text]):
        parent_state = states[parent_names[index]]
        return parent_state[keys.STATES.ENERGY], parent_state[keys.STATES.LATENT]
    return _f_parent_fixed_index


def f_child_sample_factory(beta: ts.Float = 0):
    """
    Args:
        beta: softmax temperature. Higher means more random

    Returns:
        function `f_child_sample`
    """
    def f_child_sample(targets: List[Tuple[tf.Tensor, ts.NestedTensor]]):
        # select bias for top_down attention
        dist = tfd.OneHotCategorical(logits=[beta-energy for energy, latent in targets])
        sample = dist.sample()
        ind = tf.argmax(sample)
        energy, latent = targets[ind]
        energy = energy + dist.entropy() + -dist.log_prob(sample)
        return energy, latent

    return f_child_sample
