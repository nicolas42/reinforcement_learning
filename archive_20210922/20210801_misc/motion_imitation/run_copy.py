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



# 
# python motion_imitation/run_copy.py
# 

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

# from stable_baselines.common.callbacks import CheckpointCallback


from motion_imitation.robots import robot_config
from motion_imitation.robots import laikago





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




def build_imitation_env2(motion_files, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering,
                        robot_class=laikago.Laikago,
                        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND)):
  
  assert len(motion_files) > 0

  curriculum_episode_length_start = 20
  curriculum_episode_length_end = 600
  
  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
  ]
  print("WOOOOOOOOOO")
  print(sensors, sensors[0])

  task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)

  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            env_randomizers=randomizers, robot_sensors=sensors, task=task)

  print(env._get_observation())
  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  print(env.observation_space)
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                       trajectory_generator=trajectory_generator)
  print(env.observation_space)

  if mode == "test":
      curriculum_episode_length_start = curriculum_episode_length_end

  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=30000000,
                                                  num_parallel_envs=num_parallel_envs)
  print(env.observation_space)
  return env






def main():

  # env = build_laikago_env( 
  #     motor_control_mode = robot_config.MotorControlMode.POSITION, 
  #     enable_rendering=True)


  # policy_kwargs = {
  #     "net_arch": [{"pi": [512, 256],
  #                   "vf": [512, 256]}],
  #     "act_fun": tf.nn.relu
  # }

  # model = pposgd_simple.PPO1(
  #   policy = FeedForwardPolicy, 
  #   env = env, 
  #   verbose=1, 
  #   policy_kwargs=policy_kwargs
  # )





 
  num_procs = MPI.COMM_WORLD.Get_size()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

  env = build_imitation_env2(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                        num_parallel_envs=num_procs,
                                        mode="train",
                                        enable_randomizer=False,
                                        enable_rendering=True)
  


  # Model

  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
  }


  timesteps_per_actorbatch = int(np.ceil(float(4096) / num_procs))
  optim_batchsize = int(np.ceil(float(256) / num_procs))

  # from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn, mlp_extractor, linear

  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=timesteps_per_actorbatch,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-5,
               optim_batchsize=optim_batchsize,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant',
               policy_kwargs=policy_kwargs,
               tensorboard_log="output",
               verbose=1)




  model.load_parameters("motion_imitation/data/policies/dog_pace.zip")




  # env.set_ground(env._pybullet_client.loadURDF("plane_implicit_modified.urdf"))

  observation = env.reset()
  while True:
    # action = laikago.INIT_MOTOR_ANGLES
    # The action is essentially the target position
    # to use a model. pass it the current state (observation) and it will predict an action, (and a modified state?)

    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    # print("observation,reward,done,action\n", observation,"\n", reward,"\n", done,"\n", action,"\n")
    # time.sleep(0.1)
    # if done:
    #     observation = env.reset()
    #     print("DONE")
    #     break
  
  return


if __name__ == '__main__':
  main()
