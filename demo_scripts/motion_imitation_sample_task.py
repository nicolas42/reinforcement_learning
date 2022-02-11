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


class SimpleForwardTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    # self.time_steps = 0

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    return rot_mat[-1] < 0.85


  def reward(self, env):
    """
    Get the reward without side effects.
    0.37 seems to be the height of the robot when it's lying down.  Don't want it to lie down that's not productive
    r = (self.current_base_pos[2] - 0.37) # maximize height)

    """
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    
    rotation_reward = (rot_mat[-1] - 0.85) # minimize rotations

    height_reward = 0
    if ( self.current_base_pos[2] > 0.45 ): height_reward = 0.1 # don't lie down
    
    del env
    # print( height_reward, rotation_reward )
    print( self.current_base_pos )
    return height_reward + rotation_reward
  
  


# get bonus for legs being in the same position
# or having symmetry
# orientation similarity to previous step
# maintaining height at 0.5

# remove points for energy usage.  i.e. the desired position and the current position
# being far away from each othere

# get points based on length of episode - think this is already a thing
# points if joint anglees are the same
# points if joint angle previous is similar to joint angle next
# points if you stay at a reasonable height ( normal distribution )
# ??? points if body orientation is similar to previous body orientation
# points if you move forward


