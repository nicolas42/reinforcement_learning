
import sys 
import os 
sys.path.append(os.getcwd()) # python syspath sucks

import tensorflow as tf
from motion_imitation.robots import robot_config
from motion_imitation.envs import env_builder as env_builder
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

import wandb

wandb.init(project='test', entity='nick4235')
config = wandb.config

env = env_builder.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True)

model = PPO1(MlpPolicy, env, verbose=1, policy_kwargs = {
    "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
    "act_fun": tf.nn.relu
})

model.learn(total_timesteps=1000e3)
model.save("my_ppo_1000e3_timesteps")

model.load_parameters("my_ppo_1000e3_timesteps")
# model = PPO1.load("omg") # Loading a model without an environment, this model cannot be trained until it has a valid environment.
o = env.reset()
while True:
    a, _ = model.predict(o, deterministic=False)
    print("\n\n\nobservation, action\n", o,"\n",a)
    o, r, done, info = env.step(a)
    if done:
        o = env.reset()
