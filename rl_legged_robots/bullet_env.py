import time
import math
import random
import numpy as np
import gym
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import sys
import os
from robots import Hexapod, Minitaur


class HexapodBulletEnv(gym.Env):
    def __init__(self, action_space=30, obs_space=60, seed=None, robot='hexapod', render=False):

        if(isinstance(action_space, int)):
            self.action_space = gym.spaces.box.Box(
                low=np.zeros((1, action_space)),
                high=np.ones((1, action_space))
            )
        elif(isinstance(action_space, gym.spaces.box.Box)):
            self.action_space = action_space
        else:
            raise ValueError('Action space must be a gym Box at this time or an int of the box\'s dimensions.')

        if(isinstance(obs_space, int)):
            self.obs_space = gym.spaces.box.Box(
                low=np.zeros((1, obs_space)),
                high=np.ones((1, obs_space))
            )
        elif(isinstance(obs_space, gym.spaces.box.Box)):
            self.obs_space = obs_space
        else:
            raise ValueError('Observation space must be a gym Box at this time or an int of the box\'s dimensions.')

        self.rand, self.seed = gym.utils.seeding.np_random(seed)

        if render:
            client_type = p.GUI
        else:
            client_type=p.DIRECT

        self.client = p.connect(client_type)
        self.rendered_image = None
        self.robot_type = robot
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(0.001, self.client)
        self.sim_complete = False
        self.reset()
        self.num_steps = 0

    def reset(self):
        """
        You will need to redefine this function to load in custem terrain types in each simulation episode
        """
        
        p.resetSimulation(self.client)
        p.loadURDF('plane.urdf')
        p.setGravity(0, 0, -9.81)
        if self.robot_type == 'hexapod':
            self.robot = Hexapod(self.client)
            self.observation = self.robot.observe()
        elif self.robot_type == 'minitaur':
            self.robot = Minitaur(urdfRootPath='robots/minitaur/minitaur_v1.urdf')
        for i in range(100):
            _ = p.stepSimulation()
        self.sim_complete = False
        
        return self.observation

    def step(self, action):
        initial_pos = self.robot.get_body_state()
        self.robot.act(action)
        for i in range(5):
            _ = p.stepSimulation()
        
        final_pos = self.robot.get_body_state()
        #calculate reward. You will have tochange this to your desired reward function
        reward = 0
        self.num_steps += 1
        self.sim_complete = False
        return self.robot.observe(), reward, self.sim_complete, {}

    def render(self):
        if self.rendered_image is None:
            self.rendered_image = plt.imshow(np.zeros((480, 640, 4)))
            self.projection = p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100)

        pos, rot = p.getBasePositionAndOrientation(self.robot.robot, self.client)
        # Rotate camera direction and shift up
        pos=list(pos)
        pos[2] += 0.2
        rot_mat = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(480, 640, view_matrix, view_matrix)[2]
        frame = np.reshape(frame, (480, 640, 4))
        self.rendered_image.set_data(frame)
        plt.draw()
        plt.pause(.00001)


    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.rand, self.seed = gym.utils.seeding.np_random(seed)
        return self.seed

