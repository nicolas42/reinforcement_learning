# python motion_imitation/run_copy3.py


# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True



from stable_baselines.common.base_class import BaseRLModel
import warnings

def load_parameters(self, load_path_or_dict, exact_match=True):
    """
    Load model parameters from a file or a dictionary

    Dictionary keys should be tensorflow variable names, which can be obtained
    with ``get_parameters`` function. If ``exact_match`` is True, dictionary
    should contain keys for all model's parameters, otherwise RunTimeError
    is raised. If False, only variables included in the dictionary will be updated.

    This does not load agent's hyper-parameters.

    .. warning::
        This function does not update trainer/optimizer variables (e.g. momentum).
        As such training after using this function may lead to less-than-optimal results.

    :param load_path_or_dict: (str or file-like or dict) Save parameter location
        or dict of parameters as variable.name -> ndarrays to be loaded.
    :param exact_match: (bool) If True, expects load dictionary to contain keys for
        all variables in the model. If False, loads parameters only for variables
        mentioned in the dictionary. Defaults to True.
    """
    # Make sure we have assign ops
    if self._param_load_ops is None:
        self._setup_load_operations()

    if isinstance(load_path_or_dict, dict):
        # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
        params = load_path_or_dict
    elif isinstance(load_path_or_dict, list):
        warnings.warn("Loading model parameters from a list. This has been replaced " +
                      "with parameter dictionaries with variable names and parameters. " +
                      "If you are loading from a file, consider re-saving the file.",
                      DeprecationWarning)
        # Assume `load_path_or_dict` is list of ndarrays.
        # Create param dictionary assuming the parameters are in same order
        # as `get_parameter_list` returns them.
        params = dict()
        for i, param_name in enumerate(self._param_load_ops.keys()):
            params[param_name] = load_path_or_dict[i]
    else:
        # Assume a filepath or file-like.
        # Use existing deserializer to load the parameters.
        # We only need the parameters part of the file, so
        # only load that part.
        _, params = BaseRLModel._load_from_file(load_path_or_dict, load_data=False)
        params = dict(params)

    feed_dict = {}
    param_update_ops = []
    # Keep track of not-updated variables
    not_updated_variables = set(self._param_load_ops.keys())
    for param_name, param_value in params.items():
        placeholder, assign_op = self._param_load_ops[param_name]
        feed_dict[placeholder] = param_value
        # Create list of tf.assign operations for sess.run
        param_update_ops.append(assign_op)
        # Keep track which variables are updated
        not_updated_variables.remove(param_name)

    # Check that we updated all parameters if exact_match=True
    if exact_match and len(not_updated_variables) > 0:
        raise RuntimeError("Load dictionary did not contain all variables. " +
                            "Missing variables: {}".format(", ".join(not_updated_variables)))


    # print(param_update_ops, feed_dict)
    # self.sess.run(param_update_ops, feed_dict=feed_dict)
# 
# This causes the following error
# 
# Traceback (most recent call last):
#   File "motion_imitation/run_copy3.py", line 292, in <module>
#     main()
#   File "motion_imitation/run_copy3.py", line 279, in main
#     load_parameters(model, "motion_imitation/data/policies/dog_pace.zip", exact_match = False)
#   File "motion_imitation/run_copy3.py", line 213, in load_parameters
#     self.sess.run(param_update_ops, feed_dict=feed_dict)
#   File "/home/sch600/miniconda3/envs/motion_imitation/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 956, in run
#     run_metadata_ptr)
#   File "/home/sch600/miniconda3/envs/motion_imitation/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1156, in _run
#     (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
# ValueError: Cannot feed value of shape (160, 512) for Tensor 'Placeholder:0', which has shape '(28, 512)'


    


import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import locomotion_gym_env
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.envs.env_wrappers import imitation_wrapper_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from motion_imitation.envs.env_wrappers import trajectory_generator_wrapper_env
from motion_imitation.envs.env_wrappers import simple_openloop
from motion_imitation.envs.env_wrappers import simple_forward_task
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import default_task

from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import sensor_wrappers
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
from motion_imitation.robots import laikago
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config





def build_laikago_env( motor_control_mode, enable_rendering):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  
  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = laikago.Laikago

  sensors = [
      robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS),
      robot_sensors.IMUSensor(),
      environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS)
  ]

  task = default_task.DefaultTask() # simple_forward_task.SimpleForwardTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            robot_sensors=sensors, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                      trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND))

  return env




def main():


  
  num_procs = MPI.COMM_WORLD.Get_size()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  
  # env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
  #                                       num_parallel_envs=1,
  #                                       mode="test",
  #                                       enable_randomizer=False,
  #                                       enable_rendering=True)





  env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True)

  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
  }

  import gym
  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines import PPO1
  model = PPO1(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs)



  # timesteps_per_actorbatch = int(np.ceil(float(TIMESTEPS_PER_ACTORBATCH) / num_procs))
  # optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / num_procs))
  # model = ppo_imitation.PPOImitation(
  #              policy=imitation_policies.ImitationPolicy,
  #              env=env,
  #              gamma=0.95,
  #              timesteps_per_actorbatch=timesteps_per_actorbatch,
  #              clip_param=0.2,
  #              optim_epochs=1,
  #              optim_stepsize=1e-5,
  #              optim_batchsize=optim_batchsize,
  #              lam=0.95,
  #              adam_epsilon=1e-5,
  #              schedule='constant',
  #              policy_kwargs=policy_kwargs,
  #              tensorboard_log="output",
  #              verbose=1)
  

  load_parameters(model, "motion_imitation/data/policies/dog_pace.zip", exact_match = False)


  # print(model.get_env(),  model.get_parameter_list())

# ValueError: Cannot feed value of shape (160, 512) for Tensor 'Placeholder:0', which has shape '(28, 512)'

# <motion_imitation.envs.env_wrappers.trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv object at 0x7f6136285050> [<tf.Variable 'model/pi_fc0/w:0' shape=(28, 512) dtype=float32_ref>, <tf.Variable 'model/pi_fc0/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'model/vf_fc0/w:0' shape=(28, 512) dtype=float32_ref>, <tf.Variable 'model/vf_fc0/b:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'model/pi_fc1/w:0' shape=(512, 256) dtype=float32_ref>, <tf.Variable 'model/pi_fc1/b:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'model/vf_fc1/w:0' shape=(512, 256) dtype=float32_ref>, <tf.Variable 'model/vf_fc1/b:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'model/vf/w:0' shape=(256, 1) dtype=float32_ref>, <tf.Variable 'model/vf/b:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'model/pi/w:0' shape=(256, 12) dtype=float32_ref>, <tf.Variable 'model/pi/b:0' shape=(12,) dtype=float32_ref>, <tf.Variable 'model/pi/logstd:0' shape=(1, 12) dtype=float32_ref>, <tf.Variable 'model/q/w:0' shape=(256, 12) dtype=float32_ref>, <tf.Variable 'model/q/b:0' shape=(12,) dtype=float32_ref>]

  observation = env.reset()
  print("env.observation_space", env.observation_space)
  print("model.observation_space", model.observation_space)

  # while True:
  #   action = laikago.INIT_MOTOR_ANGLES
  #   # action, _ = model.predict(observation, deterministic=True)
  #   print("action\n", action)
  #   observation, reward, done, info = env.step(action)



if __name__ == '__main__':
  main()
