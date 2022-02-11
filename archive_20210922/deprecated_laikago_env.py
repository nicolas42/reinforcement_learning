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
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import robot_config
from motion_imitation.robots import laikago

from motion_imitation.envs import locomotion_gym_env
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.envs.env_wrappers import default_task
from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import robot_sensors

from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation


# The laikago gym environment is specified here.
# This function comes from the env_builder.py file
def env_builder_build_laikago_env( motor_control_mode, enable_rendering):

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

  task = default_task.DefaultTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class, # <---
                                            robot_sensors=sensors, task=task)
  return env





def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
  }

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  # from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn, mlp_extractor, linear
  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy, # FeedForwardPolicy ???
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
               tensorboard_log=output_dir,
               verbose=1)
  return model



def main():


  # # Make a Gym Environment.  This is from motion_imitation/envs/locomotion_gym_env.py
  env = env_builder_build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True)

  
  # env = env_builder.build_imitation_env(motion_files=["dog_pace.txt"],
  #                                       num_parallel_envs=1,
  #                                       mode="test",
  #                                       enable_randomizer=False,
  #                                       enable_rendering=True)
  





  # model = build_model(env=env,
  #                     num_procs=1,
  #                     timesteps_per_actorbatch=4096,
  #                     optim_batchsize=256,
  #                     output_dir="output")


  # model.load_parameters("dog_pace.zip")







  # env.set_ground(env._pybullet_client.loadURDF("plane_implicit_modified.urdf"))

  observation = env.reset()
  while 1:
    action = laikago.INIT_MOTOR_ANGLES
    # The action is essentially the target position
    # to use a model. pass it the current state (observation) and it will predict an action, (and a modified state?)

    # action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    print("observation,reward,done,action\n", observation,"\n", reward,"\n", done,"\n", action,"\n")
    # time.sleep(0.1)
    if done:
        observation = env.reset()
        print("DONE")
        break
  
  return

if __name__ == '__main__':
  main()






# # Notes - mostly a summary of data structures
# # ------------------------------------------------------------------------------

# IMU = inertial measurement unit
# ["R", "P", "dR", "dP"]
# roll pitch roll_speed pitch_speed
# I assume in radians and radians per second


# # Example Data from env.step(action)

#  Observation
# -------------------------------------------
#  OrderedDict([('IMU', array([ 0.0004296 ,  0.00631958,  0.00301328, -0.00729477])), 
# ('LastAction', array([ 0.  ,  0.67, -1.25,  0.  ,  0.67, -1.25,  0.  ,  0.67, -1.25,0.  ,  0.67, -1.25])), 
# ('MotorAngle', array([ 3.11013110e-04,  6.84184602e-01, -1.28844090e+00,  
# 3.10951209e-04, 6.83897403e-01, -1.28778980e+00,  
# 1.55772434e-04,  6.76567213e-01, -1.27497657e+00,  
# 1.55717217e-04,  6.76313005e-01, -1.27438347e+00]))]) 



#  Reward
# -------------------------------------------
# The reward appears to be a number from 0..1.  In the current mode (as of writing this) the reward returned seems to always be equal to 1.
#  1 



#  Done
# -------------------------------------------------
#  False




#  Action
# ------------------------------------------------
# The action is a series of four sets of three angles which correspond to the four limbs of the quadruped.  
# The three angles are the abduction angle, the hip angle, and the knee angle.
#  [ 0.    0.67 -1.25  0.    0.67 -1.25  0.    0.67 -1.25  0.    0.67 -1.25] 

# # Example Action
# # the action is a set of angles 
# # Bases on the readings from Laikago's default pose.
# INIT_MOTOR_ANGLES = np.array([
#     laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
#     laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
#     laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
# ] * NUM_LEGS)



# Pybullet terrain data is located here
# /home/sch600/miniconda3/envs/motion_imitation/lib/python3.7/site-packages/pybullet_data
# urdf files reference obj files and png files are also used.  
# so far I haven't been able to copy necessary files directly

