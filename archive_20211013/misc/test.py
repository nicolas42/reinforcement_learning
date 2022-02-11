import sys 

import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1


from motion_imitation.robots import robot_config
# from motion_imitation.envs import env_builder as env_builder
import env_builder as env_builder
import os
import datetime


# global_policy_kwargs = {
#     "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
#     "act_fun": tf.nn.relu
# }


if __name__ == '__main__':

    model_file = "output/latest.zip"

    if len(sys.argv) == 2:
        model_file = sys.argv[2]
   

    # params taken from stable baselinese PPO1 run_robotics.py
    env = env_builder.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True)
    model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                    optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear') # tensorboard_log="tensorboard_log"

    if model_file:
        model.load_parameters(model_file)


    observation = env.reset()
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, r, done, info = env.step(action)
        if done:
            observation = env.reset()



