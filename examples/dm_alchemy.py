from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, "/home/jacob/PycharmProjects/info-nodes/")

import info_nodes

import tensorflow as tf
import tf_agents as tfa
"""from tf_agents.environments import suite_gym

env = suite_gym.load('CartPole-v0')
#env = tfa.environments.tf_py_environment.TFPyEnvironment(env)

print('time step spec', env.time_step_spec())
print('act spec', env.action_spec())"""

import dm_alchemy

LEVEL_NAME = ('alchemy/perceptual_mapping_'
              'randomized_with_rotation_and_random_bottleneck')
settings = dm_alchemy.EnvironmentSettings(seed=123, level_name=LEVEL_NAME)
env = dm_alchemy.load_from_docker(settings)



