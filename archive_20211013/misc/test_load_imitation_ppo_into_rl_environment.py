# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time
# import gym
import stable_baselines as sb
# from stable_baselines.common.callbacks import CheckpointCallback
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import PPO1


from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.robots import robot_config



# Load Imitation Model
# -------------------------------------

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True


num_procs = MPI.COMM_WORLD.Get_size()
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

args_mode = "test"
args_motion_file = "motion_imitation/data/motions/dog_pace.txt"
args_output_dir = "output"
enable_env_rand = ENABLE_ENV_RANDOMIZER and (args_mode != "test")

imitation_environment = env_builder.build_imitation_env(motion_files=[args_motion_file],
    num_parallel_envs=num_procs,
    mode=args_mode,
    enable_randomizer=enable_env_rand,
    enable_rendering=False)
    
timesteps_per_actorbatch = int(np.ceil(float(TIMESTEPS_PER_ACTORBATCH) / num_procs))
optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / num_procs))

imitation_model = ppo_imitation.PPOImitation(
    policy=imitation_policies.ImitationPolicy,
    env=imitation_environment,
    gamma=0.95,
    timesteps_per_actorbatch=timesteps_per_actorbatch,
    clip_param=0.2,
    optim_epochs=1,
    optim_stepsize=1e-5,
    optim_batchsize=optim_batchsize,
    lam=0.95,
    adam_epsilon=1e-5,
    schedule='constant',
    policy_kwargs= {
            "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
            "act_fun": tf.nn.relu
    },
    tensorboard_log=args_output_dir,
    verbose=1)


imitation_model.load_parameters("motion_imitation/data/policies/dog_pace.zip")



laikago_env = env_builder.build_laikago_env( 
    motor_control_mode = robot_config.MotorControlMode.POSITION, 
    enable_rendering=False
)

data = {
    "gamma": imitation_model.gamma,
    "timesteps_per_actorbatch": imitation_model.timesteps_per_actorbatch,
    "clip_param": imitation_model.clip_param,
    "entcoeff": imitation_model.entcoeff,
    "optim_epochs": imitation_model.optim_epochs,
    "optim_stepsize": imitation_model.optim_stepsize,
    "optim_batchsize": imitation_model.optim_batchsize,
    "lam": imitation_model.lam,
    "adam_epsilon": imitation_model.adam_epsilon,
    "schedule": imitation_model.schedule,
    "verbose": imitation_model.verbose,
    "policy": imitation_model.policy,
    "observation_space": laikago_env.observation_space,
    "action_space": laikago_env.action_space,
    "n_envs": imitation_model.n_envs,
    "n_cpu_tf_sess": imitation_model.n_cpu_tf_sess,
    "seed": imitation_model.seed,
    "_vectorize_action": imitation_model._vectorize_action,
    "policy_kwargs": imitation_model.policy_kwargs
}

imitation_params = imitation_model.get_parameters()
imitation_params['model/pi_fc0/w:0'] = imitation_params['model/pi_fc0/w:0'][0:28]
imitation_params['model/vf_fc0/w:0'] = imitation_params['model/pi_fc0/w:0'][0:28]



imitation_model._save_to_file("zomg", data=data, params=imitation_params, cloudpickle=False)


rl_model = sb.PPO1(sb.common.policies.MlpPolicy, laikago_env, verbose=1, 
    policy_kwargs = {
                "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
                "act_fun": tf.nn.relu
    }
)

# PPO1.load_parameters can take a dictionary or a filename to load from :)
print(rl_model.get_parameters()['model/pi_fc0/w:0'][0][0])
rl_model.load_parameters("zomg", exact_match=False)
print(rl_model.get_parameters()['model/pi_fc0/w:0'][0][0])


# # There is a logstd layer in the stable_baselines PPO1 implementation
# # 'model/pi/logstd:0'             (1, 12)
# # which is not in imitation PPO network.
# # so we need exact_match=False





# rl_params = rl_model.get_parameters()

# for key in imitation_params.keys():
#     print(key, imitation_params[key].shape)

# for key in rl_params.keys():
#     print(key, rl_params[key].shape)



# for key in ['model/pi_fc0/w:0', 'model/pi_fc0/b:0', 'model/vf_fc0/w:0', 'model/vf_fc0/b:0', 'model/pi_fc1/w:0', 'model/pi_fc1/b:0', 'model/vf_fc1/w:0', 'model/vf_fc1/b:0', 'model/vf/w:0', 'model/vf/b:0', 'model/pi/w:0', 'model/pi/b:0', 'model/q/w:0', 'model/q/b:0']:
#     print(key, rl_params[key].shape, imitation_params[key].shape)












# from collections import OrderedDict, deque

# parameters = rl_model.get_parameter_list()
# parameter_values = rl_model.sess.run(parameters)
# return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
