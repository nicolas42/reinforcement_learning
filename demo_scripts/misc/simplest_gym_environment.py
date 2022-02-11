import numpy as np
import gym
import random

class BasicEnv(gym.Env):

    def __init__(self):
        # There are two actions, first will get reward of 1, second reward of -1. 
#         self.action_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(2)

        print(self.action_space)
        print(self.observation_space)
        
        # self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        # self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def step(self, action):

        # if we took an action, we were in state 1
        state = 1
    
        if action == 2:
            reward = 1
        else:
            reward = -1
            
        # regardless of the action, game is done after a single step
        done = True

        info = {}

        return state, reward, done, info

    def reset(self):
        state = 0
        return state





env = BasicEnv()


from stable_baselines3.common.env_checker import check_env

ret = check_env(env)
print(ret)
print("check_env returns none if everything is okay")


