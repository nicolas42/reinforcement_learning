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
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math 

class MyTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.last_motor_angles = np.zeros(12)
    self.current_motor_angles = np.zeros(12)
    self.last_base_orientation = np.zeros(4)
    self.current_base_orientation = np.zeros(4)

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    # print("woo")
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
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    # print(rot_mat)
    return rot_mat[-1] < 0.85

    # my more lenient done function
    # rot_quat = env.robot.GetBaseOrientation()
    # euler_angles = env.pybullet_client.getEulerFromQuaternion(rot_quat)
    # print(euler_angles)
    # diff_angles = np.array(euler_angles) - np.array([math.pi / 2.0, 0, math.pi / 2.0])
    # return abs(diff_angles[0]) > (math.pi / 2)



  def reward(self, env):
    """Get the reward without side effects."""

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
    V = np.exp(-5*(base_velocity[0]-1)**2)-0.02

    # Maintain Orientation
    # rot_quat = env.robot.GetBaseOrientation()
    # position, orientation = pyb.getBasePositionAndOrientation(uid)
    # orientation = pyb.getEulerFromQuaternion(orientation)
    # diff_orientation = np.array(orientation) - np.array([math.pi / 2.0, 0, math.pi / 2.0])
    # O = np.exp(-np.dot(diff_orientation, diff_orientation))
    # initial_orientation = np.array([math.pi / 2.0, 0, math.pi / 2.0])

    # transform z euler angle rotation so it starts at zero when robot is facing the +x direction
    position, orientation = pyb.getBasePositionAndOrientation(uid)
    orientation = pyb.getEulerFromQuaternion(orientation)
    rot = orientation[2]
    rot -= math.pi/2 
    if rot < -math.pi:
      rot += 2*math.pi 
    # print(rot)

    # Orientation reward
    O = 2 * np.exp(-rot**2)-0.05
  

    print("energy: %+.2f \t height: %+.2f \t orientation: %+.2f, \t\t\t NOT INCLUDED: velocity: %+.2f, " % (E,H,O,V) )
    
    reward = E+H+O
    del env
    return reward

    # h = max(0, np.exp(-100*(z-0.42)**2) ) 
    # H = 2 * ( 1/(10*abs(z-0.42)+1) -1/2) 
    # spikey height reward at 0.42 which is negative outside of 0.3..0.5



# class SimpleForwardTask(object):
#   """Default empy task."""
#   def __init__(self):
#     """Initializes the task."""
#     self.current_base_pos = np.zeros(3)
#     self.last_base_pos = np.zeros(3)

#   def __call__(self, env):
#     return self.reward(env)

#   def reset(self, env):
#     """Resets the internal state of the task."""
#     self._env = env
#     self.last_base_pos = env.robot.GetBasePosition()
#     self.current_base_pos = self.last_base_pos

#   def update(self, env):
#     """Updates the internal state of the task."""
#     self.last_base_pos = self.current_base_pos
#     self.current_base_pos = env.robot.GetBasePosition()

#   def done(self, env):
#     """Checks if the episode is over.

#        If the robot base becomes unstable (based on orientation), the episode
#        terminates early.
#     """
#     rot_quat = env.robot.GetBaseOrientation()
#     rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
#     return rot_mat[-1] < 0.85


#   def reward(self, env):
#     """Get the reward without side effects."""
#     del env
#     reward = self.current_base_pos[0] - self.last_base_pos[0]
#     return reward
