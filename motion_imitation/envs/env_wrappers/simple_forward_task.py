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
    """Get the reward without side effects."""

    robot = env.robot
    sim_model = robot.quadruped
    # ref_model = self._ref_model
    pyb = env._pybullet_client

    # root_vel_ref, root_ang_vel_ref = pyb.getBaseVelocity(ref_model)
    root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)
    # root_vel_ref = np.array(root_vel_ref)
    # root_ang_vel_ref = np.array(root_ang_vel_ref)
    root_vel_sim = np.array(root_vel_sim)
    root_ang_vel_sim = np.array(root_ang_vel_sim)

    # root_vel_diff = root_vel_ref - root_vel_sim
    # root_vel_err = root_vel_diff.dot(root_vel_diff)

    # root_ang_vel_diff = root_ang_vel_ref - root_ang_vel_sim
    # root_ang_vel_err = root_ang_vel_diff.dot(root_ang_vel_diff)

    # root_velocity_err = root_vel_err + 0.1 * root_ang_vel_err
    # root_velocity_reward = np.exp(-self._root_velocity_err_scale *
    #                               root_velocity_err)

    del env
    reward = self.current_base_pos[0] - self.last_base_pos[0]

    return reward
