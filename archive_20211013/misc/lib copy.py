
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


import importlib
# import calculate_reward



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
    print("\n\n\nWOO\n\n\n")
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
    # print(rot_mat)
    if rot_mat[-1] < 0.85:
      return 1

    # my more lenient done function
    # rot_quat = env.robot.GetBaseOrientation()
    # euler_angles = env.pybullet_client.getEulerFromQuaternion(rot_quat)
    # print(euler_angles)
    # diff_angles = np.array(euler_angles) - np.array([math.pi / 2.0, 0, math.pi / 2.0])
    # return abs(diff_angles[0]) > (math.pi / 2)


    # terminate if deviates more than 90 degrees
    # # transform z euler angle rotation so it starts at zero when robot is facing the +x direction
    # uid = env.robot.quadruped
    # pyb = env._pybullet_client
    # position, orientation = pyb.getBasePositionAndOrientation(uid)
    # orientation = pyb.getEulerFromQuaternion(orientation)
    # rot = orientation[2]
    # rot -= math.pi/2 
    # if rot < -math.pi:
    #   rot += 2*math.pi 
    # if abs(rot) > math.pi/4:
    #     return 1
    
    return 0

  def reward(self, env):
    """Get the reward without side effects."""
    return calculate_reward(self,env)

    uid = env.robot.quadruped
    pyb = env._pybullet_client

    # root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(uid)
    # reward = np.array(root_vel_sim).dot(np.array([1,0,0]))

    base_velocity,_ = pyb.getBaseVelocity(uid)
    z = self.current_base_pos[2]
    diff_motor_angles = self.current_motor_angles-self.last_motor_angles
    diff_position = np.array(self.current_base_pos)-np.array(self.last_base_pos)
    diff_orientation = np.array(self.current_base_orientation)-np.array(self.last_base_orientation)
    
    # Stay still
    # seems like a valuable curriculum
    # reward = np.exp(-10*(np.dot(self.current_base_pos,self.current_base_pos)))

    # minimize energy v1
    # minimize energy through minimizing differences of motor angles
    # energy_reward = np.exp(-25*np.dot(diff_motor_angles,diff_motor_angles))
    # motor angle differences range 0..0.2 and there are 12 of them.    
    
    # energy, height, velocity, orientation, symmetry

    # Minimize Energy
    E = 10 * np.exp(-env.robot.GetEnergyConsumptionPerControlStep())

    # Height
    H = np.exp(-10000*(z-0.45)**4)-0.05
    # inspection of mocap indicates z between 0.4 and 0.5 is good.
    # z~=0.2 is lying down which we don't want.
    # this provides a reasonably square reward for z above 0.3 and below 0.6
    # outside of this it is slightly negative

    # Velocity: 
    # aim to go 1 m/s
    # big normal distribution which goes slightly negative outside of 0..1.8 m/s
    V = np.exp(-50*(base_velocity[0]-1)**2)-0.02

    # Maintain Orientation
    # rot_quat = env.robot.GetBaseOrientation()
    # position, orientation = pyb.getBasePositionAndOrientation(uid)
    # orientation = pyb.getEulerFromQuaternion(orientation)
    # diff_orientation = np.array(orientation) - np.array([math.pi / 2.0, 0, math.pi / 2.0])
    # O = np.exp(-np.dot(diff_orientation, diff_orientation))
    # initial_orientation = np.array([math.pi / 2.0, 0, math.pi / 2.0])

    # Orientation
    # transform z euler angle rotation so it starts at zero when robot is facing the +x direction
    position, orientation = pyb.getBasePositionAndOrientation(uid)
    orientation = pyb.getEulerFromQuaternion(orientation)
    rot = orientation[2]
    rot -= math.pi/2 
    if rot < -math.pi:
      rot += 2*math.pi 
    # print(rot)

    # Orientation reward
    O = np.exp(-5*rot**2)-0.05


    # Symmetry
    # rewards corresponding joints from different legs having similar angles
    # there's 12 motors, front right front left rear right rear left
    # hip, upper leg, lower leg
    joints = np.array(self.current_motor_angles)
    S = np.exp( -np.var(joints[[0,3,6,9]]) -np.var(joints[[1,4,7,10]]) -np.var(joints[[2,5,8,11]]) )



    print({ "energy": E, "height": H, "orientation": O, "velocity": V, "symmetry": S })
    
    reward = O
    del env
    return reward

    # h = max(0, np.exp(-100*(z-0.42)**2) ) 
    # H = 2 * ( 1/(10*abs(z-0.42)+1) -1/2) 
    # spikey height reward at 0.42 which is negative outside of 0.3..0.5





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




def calculate_reward(self, env):
  """Get the reward without side effects."""

  uid = env.robot.quadruped
  pyb = env._pybullet_client

  base_velocity,_ = pyb.getBaseVelocity(uid)
  z = self.current_base_pos[2]
  # diff_motor_angles = self.current_motor_angles-self.last_motor_angles
  # diff_position = np.array(self.current_base_pos)-np.array(self.last_base_pos)
  # diff_orientation = np.array(self.current_base_orientation)-np.array(self.last_base_orientation)

  position, orientation = pyb.getBasePositionAndOrientation(uid)
  orientation = pyb.getEulerFromQuaternion(orientation)
  yaw = orientation[2]
  yaw -= math.pi/2 
  if yaw < -math.pi:
    yaw += 2*math.pi 

  joints = np.array(self.current_motor_angles)
  # Stay still
  # pos = np.array(self.current_base_pos)
  # V = np.exp(-np.dot(pos,pos))

  # maintain stability
  # reward orientation stability
  # initial_orientation = np.array([math.pi / 2.0, 0, math.pi / 2.0])
  # try to keep within 45 degrees
  # O2 = np.exp(-np.dot( np.array(orientation) - initial_orientation))

  # reward based on distance from point?
  # reward equals change in distance

  # Energy
  E = 10*(np.exp(-env.robot.GetEnergyConsumptionPerControlStep()))
  # Height
  # H = (np.exp(-1000*(z-0.5)**4)-0.05)
  H = 1*(2*( 1/(1+np.exp(-100*(z-0.35))) -1/2 )) # sigmoid, above 0.3m is good.
  # Orientation
  # Keep z euler angle constant
  O = 1*(2*(np.exp(-3*yaw**2)-1/2)) # +-30 degrees is positive


  # Velocity: 
  V = 1*(base_velocity[0]) 
  # V = 10*(np.exp(-0.45*(base_velocity[0]-3)**2)-0.02)
  # V = 50*(1/(1+np.exp(-0.1*base_velocity[0])) - 1/2) # s-curve, max 10 reward, gradual incline
  # penalise movement in other directions
  # V = 100*( 2*(self.current_base_pos[0] - self.last_base_pos[0]) -(self.current_base_pos[1] - self.last_base_pos[1]) -(self.current_base_pos[2] - self.last_base_pos[2]) )

  
  # Symmetry
  S = 0
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
  P = 0
  # self.joint_history.append(joints)
  # print(len(self.joint_history))
  # if len(self.joint_history) >= 60:
  #   P = 1*( np.exp( -10*np.var(np.array(joints)-np.array(self.joint_history[-30]) ) ) )
  #   self.joint_history = self.joint_history[-65:]
  #   # print(joints, self.joint_history[-30], P)
  #   print(P)


  # Gravity Challenge :)
  # ---------------------------
  # pyb.setGravity(0,0,-9.8)
  if (env._env_step_counter % 10000 == 0):
    g_x = np.random.normal(scale=3)
    g_y = np.random.normal(scale=3)
    g_z = -9.8 + np.random.normal(scale=3)
    print("\n\n\nCHANGE GRAVITY",g_x, g_y, g_z, "\n\n\n")
    pyb.setGravity(g_x,g_y,g_z)

  # Apply external force
  # https://www.programcreek.com/python/example/122123/pybullet.applyExternalForce
  
  # print(base_velocity[0], V)
  print({ "energy": round(E,1), "height": round(H,1), "orientation": round(O,1), "velocity": round(V,1), "symmetry": round(S,1), "periodicity": round(P,1) })

  reward = E + H + O + S + V + P
  del env
  return reward







# def calculate_reward(self, env):
#   """Get the reward without side effects."""

#   uid = env.robot.quadruped
#   pyb = env._pybullet_client

#   base_velocity,_ = pyb.getBaseVelocity(uid)
#   z = self.current_base_pos[2]
#   diff_motor_angles = self.current_motor_angles-self.last_motor_angles
#   diff_position = np.array(self.current_base_pos)-np.array(self.last_base_pos)
#   diff_orientation = np.array(self.current_base_orientation)-np.array(self.last_base_orientation)

#   # Energy
#   E = np.exp(-env.robot.GetEnergyConsumptionPerControlStep())
#   # Height
#   H = np.exp(-10000*(z-0.45)**4)-0.05
#   # Velocity: 
#   V = np.exp(-50*(base_velocity[0]-1)**2)-0.02

#   # Orientation
#   position, orientation = pyb.getBasePositionAndOrientation(uid)
#   orientation = pyb.getEulerFromQuaternion(orientation)
#   rot = orientation[2]
#   rot -= math.pi/2 
#   if rot < -math.pi:
#     rot += 2*math.pi 
#   O = np.exp(-5*rot**2)-0.05

#   # Symmetry
#   joints = np.array(self.current_motor_angles)
#   S = np.exp( -np.var(joints[[0,3,6,9]]) -np.var(joints[[1,4,7,10]]) -np.var(joints[[2,5,8,11]]) )

#   print({ "energy": E, "height": H, "orientation": O, "velocity": V, "symmetry": S })
  
#   reward = E + H + O + V + S
#   del env
#   return reward

