from typing import Text, List, Mapping, Callable, Tuple, Union, Counter

import tensorflow as tf

from ...utils import types as ts
from ...utils import keys
from ..info_node import InfoNode
from ..info_node import functions as infonode_funcs
from . import functions


class AttnNode(InfoNode):
    """
    Generic AttnNode. Modular functions pool inputs, build bottom-up and top-down
    attention maps, and then pool attended latent values. Only the latent member
    `keys.STATES.ATTN_BIAS` is controllable.

    This InfoNode is composed of modular attention functions with kwargs:
    - f_parent: states, parent_names -> (energy, input)
    - f_child: targets -> (energy, latent_target)
    - f_bu_attn: input -> (attention_map)
    - f_td_attn: input, bias -> (attention_map)
    - f_attend: input, attention_map -> attended_value
    Function implimentations are prefixed by their purpose name (EG: `f_attend_L1`)
    and defined in `functions`. See subclasses for ready built implementations of
    these functions.

    NOTE: parents must have homogenous latent structures
    """

    _NEXT_ROUND_ENERGY = 'next_round_energy'

    def __init__(self,
                 f_parent: Callable,  # [[ts.NestedTensor, List[Text]], Tuple[tf.Tensor, ts.NestedTensor]]
                 f_child: Callable,  # [[List[Tuple[tf.Tensor, ts.NestedTensor]]], Tuple[tf.Tensor, ts.NestedTensor]]
                 f_bu_attn: Callable,
                 f_td_attn: Callable,
                 f_attend: Callable,
                 attended_value_spec: ts.NestedTensorSpec,
                 attention_bias_spec: ts.NestedTensorSpec,
                 parent_names: List[Text],
                 num_children: ts.Int,
                 weight_bu_attn: ts.Float = 1.,
                 weight_td_attn: ts.Float = 1.,
                 name: str = 'AttnNode'):
        """Public and subclass constructor"""

        latent_spec = {
            keys.STATES.ATTENTION_BIAS: attention_bias_spec,
            keys.STATES.ATTENDED_VALUE: attended_value_spec,
        }
        controllable_latent_spec = {
            keys.STATES.ATTENTION_BIAS: tf.nest.map_structure(lambda _: True, attention_bias_spec),
            keys.STATES.ATTENDED_VALUE: tf.nest.map_structure(lambda _: False, attended_value_spec),
        }

        super(AttnNode, self).__init__(
            state_spec_extras={AttnNode._NEXT_ROUND_ENERGY: tf.TensorSpec(tuple())},
            controllable_latent_spec=controllable_latent_spec,
            parent_names=parent_names,
            num_children=num_children,
            latent_spec=latent_spec,
            f_parent=f_parent,
            f_child=f_child,
            name=name
        )

        self.f_parent = f_parent
        self.f_child = f_child
        self.f_bu_attn = f_bu_attn
        self.f_td_attn = f_td_attn
        self.f_attend = f_attend

        self.weight_bu_attn = weight_bu_attn
        self.weight_td_attn = weight_td_attn

    def bottom_up(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:

        # get attention information
        parent_energy, parent_latent = self.f_parent(states=states, parent_names=self.parent_names)
        child_energy, child_target_latent = self.f_child(targets=states[self.name][keys.STATES.TARGET_LATENTS])
        # perform attention
        bottom_up_attn_map = self.f_bu_attn(input=parent_latent)
        top_down_attn_map = self.f_td_attn(input=parent_latent, bias=child_target_latent)
        attention_map = tf.nest.map_structure(
            lambda bu, td: parent_energy*self.weight_bu_attn*bu + child_energy*self.weight_td_attn*td,
            bottom_up_attn_map, top_down_attn_map)
        attended_value: ts.NestedTensor = self.f_attend(input=parent_latent, attention_map=attention_map)
        states[self.name][keys.STATES.LATENT][keys.STATES.ATTENDED_VALUE] = attended_value
        states[self.name][keys.STATES.LATENT][keys.STATES.ATTENTION_BIAS] = child_target_latent
        states[self.name][keys.STATES.ENERGY] = parent_energy + child_energy + states[self.name][AttnNode._NEXT_ROUND_ENERGY]
        return states

    def top_down(self, states: Mapping[Text, ts.NestedTensor]) -> Mapping[Text, ts.NestedTensor]:
        next_round_energy, attention_bias = self.f_child(states[self.name][keys.STATES.TARGET_LATENTS])
        states[self.name][AttnNode._NEXT_ROUND_ENERGY] = next_round_energy
        states[self.name][keys.STATES.LATENT][keys.STATES.ATTENTION_BIAS] = attention_bias
        return states

    def train(self, experience: ts.NestedTensor) -> None:
        pass


class ATTENTION_FOCUS:
    DIFFERENT = 'different'
    SAME = 'same'


class DenseAttnNode(AttnNode): # is this just 1x1x1... convolution?
    """Attends to the last dimension of dense vectors by bottom up and top down processes:
    - Bottom up attention is determined by the L1 norm of the last dimension.
    - Top down attention is determined by the LP or dot-product similarity/difference between
        a bias tensor and final dimension tensor slice.

    NOTE: parent latents must be shaped (B, ..., C) and attention bias (B, C)."""

    def __init__(self,
                 parent_latent_spec: ts.NestedTensorSpec,
                 parent_names: List[Text],
                 num_children: int,
                 td_similarity_metric: Union[int, Text] = 'dotprod',  # 'dotprod' or an integer for Lp similarity
                 td_attention_focus: Text = ATTENTION_FOCUS.DIFFERENT,
                 name='NestedDenseAttentionInfoNode'):

        f_parent = infonode_funcs.f_parent_concat  # don't change this unless you know what you're doing
        f_bu_attn = functions.f_bu_attn_LP_diff_factory(p=1)
        f_attend = functions.f_attend_softmax_pool

        if td_similarity_metric == 'dotprod':
            f_td_attn = functions.f_td_attn_dotprod_similarity
        else:
            f_td_attn = functions.f_td_attn_Lp_similarity_factory(p=td_similarity_metric)
        if td_attention_focus == ATTENTION_FOCUS.DIFFERENT:
            def inverted_attn(inputs, bias):
                return tf.nest.map_structure(f_td_attn(inputs, -bias), inputs, bias)
            f_td_attn = inverted_attn

        tmp_parent_latent = tf.nest.map_structure(tf.zeros, parent_latent_spec)
        tmp_states = {name: {keys.STATES.LATENT: tmp_parent_latent,
                             keys.STATES.ENERGY: 0} for name in parent_names}
        _, tmp_input_parent_latent = f_parent(states=tmp_states, parent_names=parent_names)
        tmp_bu_attn_map = f_bu_attn(input=tmp_input_parent_latent)
        tmp_bias = tf.nest.map_structure(lambda t: tf.zeros((t.shape[0], t.shape[-1])), tmp_input_parent_latent)
        tmp_td_attn_map = f_td_attn(input=tmp_input_parent_latent, bias=tmp_bias)  # shape checking
        tmp_attended_value = f_attend(input=tmp_parent_latent, attention_map=tmp_bu_attn_map+tmp_td_attn_map)
        attended_value_spec = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape), tmp_attended_value)
        bias_spec = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape), tmp_bias)

        super(DenseAttnNode, self).__init__(
            f_parent=f_parent,
            f_child=infonode_funcs.f_child_sample_factory(),
            f_bu_attn=f_bu_attn,
            f_td_attn=f_td_attn,
            f_attend=functions.f_attend_softmax_pool,
            attended_value_spec=attended_value_spec,
            attention_bias_spec=bias_spec,
            parent_names=parent_names,
            num_children=num_children,
            name=name)


class ConvAttnNode(AttnNode):
    """Attends to the last dimension of dense vectors by bottom up and top down processes:
    - Bottom up attention is determined by the L1 norm of the last dimension.
    - Top down attention is determined by the LP or dot-product similarity/difference between
        a bias tensor and convolutional windows over the input tensor.

    NOTE parent latents must be shaped (B,L1,L2,L3,C) and attention bias (B,w1,w2,w3,C)
    for 3D convolution (or with fewer L and w axes for 2D and 1D convolution)."""

    def __init__(self,
                 parent_latent_spec: ts.NestedTensorSpec,
                 parent_names: List[Text],
                 num_children: int,
                 td_attention_focus: Text = ATTENTION_FOCUS.DIFFERENT,
                 name='NestedDenseAttentionInfoNode'):

        f_parent = infonode_funcs.f_parent_concat  # don't change this unless you know what you're doing
        f_bu_attn = functions.f_bu_attn_LP_diff_factory(p=1)
        f_td_attn = functions.f_td_attn_conv_similarity_factory(
            ndim=tf.nest.flatten(parent_latent_spec)[0].shape.as_list()-1,
            strides=1, padding='SAME')
        f_attend = functions.f_attend_softmax_pool

        if td_attention_focus == ATTENTION_FOCUS.DIFFERENT:
            def inverted_attn(input, bias):
                return tf.nest.map_structure(f_td_attn(input=input, bias=-bias), input, bias)
            f_td_attn = inverted_attn

        tmp_parent_latent = tf.nest.map_structure(tf.zeros, parent_latent_spec)
        tmp_states = {name: {keys.STATES.LATENT: tmp_parent_latent,
                             keys.STATES.ENERGY: 0} for name in parent_names}
        _, tmp_input_parent_latent = f_parent(states=tmp_states, parent_names=parent_names)
        tmp_bu_attn_map = f_bu_attn(input=tmp_input_parent_latent)
        tmp_bias = tf.nest.map_structure(lambda t: tf.zeros((t.shape[0], t.shape[-1])), tmp_input_parent_latent)
        tmp_td_attn_map = f_td_attn(input=tmp_input_parent_latent, bias=tmp_bias)  # shape checking
        tmp_attended_value = f_attend(input=tmp_parent_latent, attention_map=tmp_bu_attn_map+tmp_td_attn_map)
        attended_value_spec = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape), tmp_attended_value)
        bias_spec = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape), tmp_bias)

        super(ConvAttnNode, self).__init__(
            f_parent=f_parent,
            f_child=infonode_funcs.f_child_sample_factory(),
            f_bu_attn=f_bu_attn,
            f_td_attn=f_td_attn,
            f_attend=functions.f_attend_softmax_pool,
            attended_value_spec=attended_value_spec,
            attention_bias_spec=bias_spec,
            parent_names=parent_names,
            num_children=num_children,
            name=name)
