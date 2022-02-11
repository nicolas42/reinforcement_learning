# Usage: 

# Train
# python run.py --train
# trains according to parameters below
# set input_file to None for a new model or to the network file you want to start from

# Demo
# python run.py <filename.zip> 
# demos network according to parameters below
# Defaults to non-stochastic behaviour


# The PPO parameters were taken from stable baselinese PPO1 run_robotics.py
# The parameters that are different are these (these are the default values)
# timesteps_per_actorbatch=256, optim_stepsize=1e-3, optim_batchsize=64 , 
# entcoeff=0.01, optim_epochs=4, 

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

import numpy as np
import math

import os
import inspect
import sys 
import os
import datetime
import numpy as np
import math 

import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

from motion_imitation.robots import robot_config

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
# from motion_imitation.envs.env_wrappers import simple_forward_task
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import default_task

from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import sensor_wrappers
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
# from motion_imitation.robots import laikago
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config

import laikago



def calculate_reward(self, env):
    uid = env.robot.quadruped
    pyb = env._pybullet_client

    base_velocity,_ = pyb.getBaseVelocity(uid)
    z = self.current_base_pos[2]

    # default_initial_orientation = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])
    position, orientation = pyb.getBasePositionAndOrientation(uid)
    orientation = pyb.getEulerFromQuaternion(orientation)
    # Euler angles
    yaw = orientation[2]
    yaw -= math.pi/2 
    if yaw < -math.pi:
      yaw += 2*math.pi 

    roll = orientation[0]
    roll -= math.pi/2 
    if roll < -math.pi:
      roll += 2*math.pi 

    joints = np.array(self.current_motor_angles)

    # Energy - avoid spasms
    # rewards small adjustments over large ones
    E = (np.exp(-env.robot.GetEnergyConsumptionPerControlStep()))
  
    # Height - don't lie down. sigmoid, above 0.3m
    # easy
    H = 1/(1+np.exp(-100*(z-0.3)))

    # # Orientation - don't go sideways
    # # max reward = 3. goes to zero at +-60 degrees. Approx 1 at +-30 degrees.
    # O = 2*(np.exp(-5*yaw**2))
    # O = 0
    # # Orientation - roll
    # # O2 = (np.exp(-10*roll**2))
    # O2 = 0

    # Velocity - go right.
    V = (base_velocity[0]) 

    P = -abs(position[1]) # don't go in y direction

    # print(base_velocity[0], V)
    # print({ "energy": round(E,1), "height": round(H,1), "roll": round(O2,1), "yaw": round(O,1), "velocity": round(V,1)})   # , "symmetry": round(S,1), "periodicity": round(P,1) })

    print(E,H,V,P)
    reward = E + H + V + P # O + V # + O2 # + S + P 
    del env
    return reward



global_policy_kwargs = {
    "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
    "act_fun": tf.nn.relu
}

def demo(input_file="output/latest.zip", deterministic=True):

  enable_rendering = True
  env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=enable_rendering)
  model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                  optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear', policy_kwargs=global_policy_kwargs) # 'linear') # tensorboard_log="tensorboard_log"
  if input_file:
      model.load_parameters(input_file)

  observation = env.reset()
  i = 0
  while True:
    i += 1
    action, _ = model.predict(observation, deterministic=deterministic)
    observation, r, done, info = env.step(action)
    # print(observation)
    if done:
        observation = env.reset()


def train(input_file=None, output_dir="output/"):
  enable_rendering = False
  env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=enable_rendering)
  # parameters taken from stable baselinese PPO1 run_robotics.py
  model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                  optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear', 
                  tensorboard_log=output_dir + "tensorboard_log", policy_kwargs=global_policy_kwargs)
  if input_file:
      model.load_parameters(input_file)

  from stable_baselines.common.callbacks import CheckpointCallback
  checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=output_dir, name_prefix='' )
  model.learn(total_timesteps=2000000, callback=checkpoint_callback) # # , log_interval=100, tb_log_name="PPO1", reset_num_timesteps=True)


def run():

  # usage:
  # run.py --demo input_file
  # run.py input_file output_dir
  # set input_file to None to start from scratch
  
  if sys.argv[1] == "--demo":
      input_file = sys.argv[2]
      demo(input_file=input_file)
  else:
      input_file = None
      output_dir = "output/20211019/" 
      os.mkdir( output_dir )
      train(input_file=input_file, output_dir=output_dir)



def apply_external_force(env):
    # Apply external force
    # https://www.programcreek.com/python/example/122123/pybullet.applyExternalForce
    uid = env.robot.quadruped
    pyb = env._pybullet_client
    pyb.applyExternalForce(uid, -1, (1,1,1), (0, 0, 0), pyb.WORLD_FRAME)

def randomize_gravity(env):
    uid = env.robot.quadruped
    pyb = env._pybullet_client

    # Gravity Challenge :)
    # ---------------------------
    # pyb.setGravity(0,0,-9.8)
    # if (env._env_step_counter % 10000 == 0):
    g_x = np.random.normal(scale=1)
    g_y = np.random.normal(scale=1)
    g_z = -9.8 #  + np.random.normal(scale=3)
    print("\n\n\nCHANGE GRAVITY",g_x, g_y, g_z, "\n\n\n")
    pyb.setGravity(g_x,g_y,g_z)


class Reward(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.last_motor_angles = np.zeros(12)
    self.current_motor_angles = np.zeros(12)
    self.last_base_orientation = np.zeros(4)
    self.current_base_orientation = np.zeros(4)
    # self.joint_history = []


  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    # this occurs at the start of each episode
    # print("\n\n\nWOO\n\n\n")
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

    self.last_motor_angles = env.robot.GetMotorAngles()
    self.current_motor_angles = self.last_motor_angles

    
    self.last_base_orientation = env.robot.GetBaseOrientation()
    self.current_base_orientation = self.last_base_orientation

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()
    
    self.last_motor_angles = self.current_motor_angles
    self.current_motor_angles = env.robot.GetMotorAngles()

    self.last_base_orientation = self.current_base_orientation
    self.current_base_orientation = env.robot.GetBaseOrientation()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    # exit if robot is unstable
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    if rot_mat[-1] < 0.85:
      return 1
    
    return 0

  def reward(self, env):
    """Get the reward without side effects."""

    return calculate_reward(self,env)
    # return quaternion_reward(self,env)


def build_laikago_env( motor_control_mode, enable_rendering):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = True
  
  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = laikago.Laikago

  sensors = [
      robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS),
      # robot_sensors.IMUSensor(), # default channels are ["R", "P", "dR", "dP"]
      robot_sensors.IMUSensor(channels=["R", "P", "Y","dR"]),
      environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS)
  ]

  task = Reward()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            robot_sensors=sensors, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
    trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND))

  return env


if __name__ == '__main__':
  run()





# Reward Notes
# ---------------------------------------------


# penalise movement in other directions
# V = 100*( 2*(self.current_base_pos[0] - self.last_base_pos[0]) -(self.current_base_pos[1] - self.last_base_pos[1]) -(self.current_base_pos[2] - self.last_base_pos[2]) )

# # Stay still
# # seems like a valuable curriculum
# # reward = np.exp(-10*(np.dot(self.current_base_pos,self.current_base_pos)))


# Symmetry
# -----------------------
# S = 0
# just hips is probably a good idea
# S = 10*(np.exp( -10*np.var( np.array([1,-1,1,-1])*np.array(joints[[0,3,6,9]]) ))) #  -np.var(joints[[1,4,7,10]]) -np.var(joints[[2,5,8,11]]) ))
# S += 10*(np.exp( -np.var( np.array([ joints[0], -joints[3]]) ))) #  -np.var(joints[[1,4,7,10]]) -np.var(joints[[2,5,8,11]]) ))
# S += 10*(np.exp( -np.var( np.array([ joints[6], -joints[9]]) )))
# print(joints[0], joints[3], joints[6], joints[9])

# Periodicity
# -----------------------
# like transformer
# append all actions taken to an array
# there's a reward associated with taking the same action as a particular number of timesteps ago
# changing the number of timesteps should change the gait frequency
# the number of timesteps ago could be fed into the observation space as a control parameter
# P = 0
# self.joint_history.append(joints)
# print(len(self.joint_history))
# if len(self.joint_history) >= 60:
#   P = 1*( np.exp( -10*np.var(np.array(joints)-np.array(self.joint_history[-30]) ) ) )
#   self.joint_history = self.joint_history[-65:]
#   # print(joints, self.joint_history[-30], P)
#   print(P)


# uid = env.robot.quadruped
# pyb = env._pybullet_client

# # root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(uid)
# # reward = np.array(root_vel_sim).dot(np.array([1,0,0]))

# base_velocity,_ = pyb.getBaseVelocity(uid)
# z = self.current_base_pos[2]
# diff_motor_angles = self.current_motor_angles-self.last_motor_angles
# diff_position = np.array(self.current_base_pos)-np.array(self.last_base_pos)
# diff_orientation = np.array(self.current_base_orientation)-np.array(self.last_base_orientation)


# # minimize energy v1
# # minimize energy through minimizing differences of motor angles
# # energy_reward = np.exp(-25*np.dot(diff_motor_angles,diff_motor_angles))
# # motor angle differences range 0..0.2 and there are 12 of them.    


    




# def make_datetime_string():
#   return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



# def train_with_dynamic_reward(input_file=None, output_dir="output/"):
#   # reloads python file periodically so you can update parameters during training dynamically.
#   # the policy schedule is kind of meaningless since a linear schedule reduces the learning rate down to zero 
#   # during number of total timesteps which is quite low in this script.  It might make sense to change the 
#   # schedule to a 'constant' learning rate.  Alternatively a newer version of this script might allow for dynamic 
#   # changing of all of these parameters during training as well.  It would be cool to increase and decrease the 
#   # learning rate dynamically.

#   env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=False)
#   # parameters taken from stable baselinese PPO1 run_robotics.py
#   model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
#                   optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear', 
#                   tensorboard_log=output_dir + "tensorboard_log")
#   if input_file:
#       model.load_parameters(input_file)

#     # def load_parameters(self, load_path_or_dict, exact_match=True):

#   for i in range(10000):
#       print("\n\n\nRELOADING REWARD\n\n\n")
#       importlib.reload(lib)
#       # randomize_gravity(env)
#       model.learn(total_timesteps=2048) # , callback=None, log_interval=100, tb_log_name="PPO1", reset_num_timesteps=True)
#       print(output_dir + str(i))
#       model.save( output_dir + str(i) )
#       model.save( output_dir + "latest" )
#       model.save("output/latest")



# def demo_latest_policy(input_file="output/latest.zip", deterministic=True):
#   # reload model input_file regularly to show the behaviour of the most recent model.
  
#   enable_rendering = True
#   reload_model = True

#   env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=enable_rendering)
#   model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
#                   optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear') # 'linear') # tensorboard_log="tensorboard_log"
#   if input_file:
#       model.load_parameters(input_file)

#   observation = env.reset()
#   i = 0
#   while True:
#     i += 1
#     if reload_model and input_file and i % 100 == 0:
#       print("\n\n\nRELOADING MODEL\n\n\n")
#       importlib.reload(lib)
#       model.load_parameters(input_file)

#     action, _ = model.predict(observation, deterministic=deterministic)
#     observation, r, done, info = env.step(action)
#     if done:
#         observation = env.reset()





# # quaternion reward 
# # with no energy, normal height, 3X quaternion dot product orientation, 2x velocity
# def quaternion_reward(self, env):
#     uid = env.robot.quadruped
#     pyb = env._pybullet_client

#     base_velocity,_ = pyb.getBaseVelocity(uid)
#     z = self.current_base_pos[2]
#     initial_orientation = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])
#     _, orientation = pyb.getBasePositionAndOrientation(uid)

#     initial_orientation = np.array(initial_orientation[1:])
#     initial_orientation = initial_orientation / np.sqrt( np.dot( initial_orientation, initial_orientation ) )

#     orientation = np.array(orientation[1:])
#     orientation = orientation / np.sqrt( np.dot( orientation, orientation ) )

#     r = {
#       "energy": 3*(np.exp(-env.robot.GetEnergyConsumptionPerControlStep())),
#       "height": 1/(1+np.exp(-100*(z-0.3))),
#       "orientation": np.dot(orientation, initial_orientation),
#       "velocity": 2*(base_velocity[0]),
#     }

#     print( [ (key, round(r[key],1)) for key in r.keys() ] )
#     reward = r["energy"] + r["height"] + r["orientation"] + r["velocity"]

#     del env
#     return reward

