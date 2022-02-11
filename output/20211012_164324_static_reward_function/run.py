import os
import inspect
import sys 
import os
import datetime
import numpy as np
import math 

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
from motion_imitation.robots import robot_config

import importlib
import lib


# global_policy_kwargs = {
#     "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
#     "act_fun": tf.nn.relu
# }


def test(input_file="output/latest.zip", deterministic=True):

  # input_file = "output/latest.zip"
  # # input_file = "output/20211011_164720_from_scratch/200.zip"
  # deterministic = True
  enable_rendering = True
  reload_model = True



  # Make environment and model - physical and mental aspects of simulation
  # !!! should probably change schedule to 'constant' the way that I've been training
  env = lib.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=enable_rendering)
  # parameters taken from stable baselinese PPO1 run_robotics.py
  model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                  optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear') # 'linear') # tensorboard_log="tensorboard_log"

  if input_file:
      model.load_parameters(input_file)

  # test
  observation = env.reset()
  i = 0
  while True:
    i += 1
    if reload_model and input_file and i % 100 == 0:
      print("\n\n\nRELOADING MODEL\n\n\n")
      importlib.reload(lib)
      model.load_parameters(input_file)

    action, _ = model.predict(observation, deterministic=deterministic)
    observation, r, done, info = env.step(action)
    # print(observation)
    if done:
        observation = env.reset()



def train(input_file=None, output_dir="output/"):

  # input_file = "output/20211011_164720_from_scratch/200.zip"
  # reload_model = True
  # deterministic = True

  # input_file = "output/latest.zip"
  enable_rendering = False
  # output_dir = "output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +"_"+ output_dir #  "_from_scratch/" 
  # os.mkdir( output_dir )
  # print(output_dir)


  # Make environment and model - physical and mental aspects of simulation
  env = lib.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=enable_rendering)
  # parameters taken from stable baselinese PPO1 run_robotics.py
  model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                  optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear', 
                  tensorboard_log=output_dir + "tensorboard_log")

  if input_file:
      model.load_parameters(input_file)

  # effectively a variable learning rate
  for i in range(10000):
      print("\n\n\nRELOADING REWARD\n\n\n")
      importlib.reload(lib)
      # lib.randomize_gravity(env)
      model.learn(total_timesteps=2048)
      print(output_dir + str(i))
      model.save( output_dir + str(i) )
      model.save( output_dir + "latest" )
      model.save("output/latest")



def run():

  # set input_file to None for a new model

  # Training
  if len(sys.argv) == 2 and sys.argv[1] == "--train":

      input_file = None # "output/latest.zip" # "output/20211011_201920_from_scratch/100.zip"
      output_dir = "output/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_static_reward_function/" 
      os.mkdir( output_dir )
      train(input_file=input_file, output_dir=output_dir)


  # Testing
  else:
      input_file = "output/latest.zip"
      deterministic=True
      test(input_file=input_file, deterministic=deterministic)




if __name__ == '__main__':
  run()

