# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation
from stable_baselines.common.callbacks import CheckpointCallback

from motion_imitation.robots import robot_config
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

def load_a_model_using_PPO1_load():
    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        


    env = env_builder.build_laikago_env( 
        motor_control_mode = robot_config.MotorControlMode.POSITION, 
        enable_rendering=True
    )

    model = PPO1(MlpPolicy, env, verbose=1, policy_kwargs = {
        "net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    })

    model.load_parameters("motion_imitation/data/policies/dog_pace.zip", exact_match = False)


    # Without exact_match = False we get this error
    # ----------------------------------------------------
    #       Traceback (most recent call last):
    #       File "motion_imitation/run2.py", line 70, in <module>
    #        main()
    #       File "motion_imitation/run2.py", line 58, in main
    #        model.load_parameters("motion_imitation/data/policies/dog_pace.zip")
    #       File "/home/nick/student-project---nicolas-schmidt/env37/lib/python3.7/site-packages/stable_baselines/common/base_class.py", line 499, in load_parameters
    #        "Missing variables: {}".format(", ".join(not_updated_variables)))
    # RuntimeError: Load dictionary did not contain all variables. Missing variables: model/pi/logstd:0


    # Then we get this error saying that the input size is wrong.
    # ------------------------------------------------------------
    # Traceback (most recent call last):
    #       File "motion_imitation/run2.py", line 80, in <module>
    #        main()
    #       File "motion_imitation/run2.py", line 58, in main
    #        model.load_parameters("motion_imitation/data/policies/dog_pace.zip", exact_match = False)
    #       File "/home/nick/student-project---nicolas-schmidt/env37/lib/python3.7/site-packages/stable_baselines/common/base_class.py", line 501, in load_parameters
    #        self.sess.run(param_update_ops, feed_dict=feed_dict)
    #       File "/home/nick/student-project---nicolas-schmidt/env37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 956, in run
    #        run_metadata_ptr)
    #       File "/home/nick/student-project---nicolas-schmidt/env37/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1156, in _run
    #        (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
    # ValueError: Cannot feed value of shape (160, 512) for Tensor 'Placeholder:0', which has shape '(28, 512)'


    print("env.observation_space", env.observation_space)
    print("model.observation_space", model.observation_space)

    obs = env.reset()
    while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

    return


"""
I'm using stable baselines PPO1 class load and save functions
class PPO1(ActorCriticRLModel):

ppo imitation calls save from within the learn function, which makes sense.
"""



TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True

def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
                "net_arch": [{"pi": [512, 256],
                                                        "vf": [512, 256]}],
                "act_fun": tf.nn.relu
    }

    timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
    optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

    model = ppo_imitation.PPOImitation(
                                        policy=imitation_policies.ImitationPolicy,
                                        env=env,
                                        gamma=0.95,
                                        timesteps_per_actorbatch=timesteps_per_actorbatch,
                                        clip_param=0.2,
                                        optim_epochs=1,
                                        optim_stepsize=1e-5,
                                        optim_batchsize=optim_batchsize,
                                        lam=0.95,
                                        adam_epsilon=1e-5,
                                        schedule='constant',
                                        policy_kwargs=policy_kwargs,
                                        tensorboard_log=output_dir,
                                        verbose=1)
    return model

def save(self, save_path, cloudpickle=False):
        data = {
                    "gamma": self.gamma,
                    "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
                    "clip_param": self.clip_param,
                    "entcoeff": self.entcoeff,
                    "optim_epochs": self.optim_epochs,
                    "optim_stepsize": self.optim_stepsize,
                    "optim_batchsize": self.optim_batchsize,
                    "lam": self.lam,
                    "adam_epsilon": self.adam_epsilon,
                    "schedule": self.schedule,
                    "verbose": self.verbose,
                    "policy": self.policy,
                    "observation_space": self.observation_space,
                    "action_space": self.action_space,
                    "n_envs": self.n_envs,
                    "n_cpu_tf_sess": self.n_cpu_tf_sess,
                    "seed": self.seed,
                    "_vectorize_action": self._vectorize_action,
                    "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()
        print(params_to_save)
        print(data)

        # self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
        return



# Data looks like this. The parameters are saved as two files, one which is the names of the layers and the other
# which I assume are the actual parameters themselves.      This could be edited.
# ----------------------------------------------------------
# {'gamma': 0.95, 'timesteps_per_actorbatch': 4096, 'clip_param': 0.2, 'entcoeff': 0.01, 'optim_epochs': 1, 'optim_stepsize': 1e-05, 'optim_batchsize': 256, 'lam': 0.95, 'adam_epsilon': 1e-05, 'schedule': 'constant', 'verbose': 1, 'policy': <class 'motion_imitation.learning.imitation_policies.ImitationPolicy'>, 'observation_space': Box(160,), 'action_space': Box(12,), 'n_envs': 1, 'n_cpu_tf_sess': 1, 'seed': None, '_vectorize_action': False, 'policy_kwargs': {'net_arch': [{'pi': [512, 256], 'vf': [512, 256]}], 'act_fun': <function relu at 0x7f557979f170>}}






def demo_saving_and_loading_of_imitation_network():

    # To do RL training using a model that has been trained using motion imitation 
    # we need to fix the observation size.  This means changing the hyperparameter
    # and also stripping the input layers of the neural network

    # build an imitation environment and PPO model and load in trained parameters
    # ---------------------------------------------------------------------------
    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    args_mode = "test"
    args_motion_file = "motion_imitation/data/motions/dog_pace.txt"
    args_visualize = True
    args_output_dir = "output"
    enable_env_rand = ENABLE_ENV_RANDOMIZER and (args_mode != "test")

    imitation_environment = env_builder.build_imitation_env(motion_files=[args_motion_file],
        num_parallel_envs=num_procs,
        mode=args_mode,
        enable_randomizer=enable_env_rand,
        enable_rendering=args_visualize)
        
    model = build_model(env=imitation_environment,
        num_procs=num_procs,
        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
        optim_batchsize=OPTIM_BATCHSIZE,
        output_dir=args_output_dir)

    model.load_parameters("motion_imitation/data/policies/dog_pace.zip")




    # modify the environment's observation space to match the RL environment
    # --------------------------------------------------
    rl_environment = env_builder.build_laikago_env( 
        motor_control_mode = robot_config.MotorControlMode.POSITION, 
        enable_rendering=False
    )

    data = {
        "gamma": model.gamma,
        "timesteps_per_actorbatch": model.timesteps_per_actorbatch,
        "clip_param": model.clip_param,
        "entcoeff": model.entcoeff,
        "optim_epochs": model.optim_epochs,
        "optim_stepsize": model.optim_stepsize,
        "optim_batchsize": model.optim_batchsize,
        "lam": model.lam,
        "adam_epsilon": model.adam_epsilon,
        "schedule": model.schedule,
        "verbose": model.verbose,
        "policy": model.policy,
        "observation_space": rl_environment.observation_space,
        "action_space": model.action_space,
        "n_envs": model.n_envs,
        "n_cpu_tf_sess": model.n_cpu_tf_sess,
        "seed": model.seed,
        "_vectorize_action": model._vectorize_action,
        "policy_kwargs": model.policy_kwargs
    }

    params = model.get_parameters()

    print(params)
    print(data)

    # print(model.get_parameters()['model/pi_fc0/w:0'].shape)
    # print(model.get_parameters()['model/pi_fc0/b:0'].shape)
    # print(model.get_parameters()['model/vf_fc0/w:0'].shape)
    # print(model.get_parameters()['model/vf_fc0/b:0'].shape)

    """
    # SHOULD I STRIP NEURONS HERE?
    """

    model._save_to_file("zomg", data=data, params=params, cloudpickle=False)

    globals().update(locals()) # use this with python interactive mode python -i <file> for a python debugger

    print("woo")









    # # Load pruned net intoRL environment
    # -----------------------------------


    env = env_builder.build_laikago_env( 
      motor_control_mode = robot_config.MotorControlMode.POSITION, 
      enable_rendering=True
    )


    model2 = PPO1(MlpPolicy, rl_environment, verbose=1, policy_kwargs = {
        "net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    })

    globals().update(locals()) # use this with python interactive mode python -i <file> for a python debugger

    # model2.learn(total_timesteps=1)
    # model2.save("zomg")
    globals().update(locals())

    print("\n\n\n\nbefore load\n\n", model2.get_parameters()['model/vf_fc1/w:0'])
    model2.load_parameters("zomg")
    print("\n\n\n\nafter load\n\n", model2.get_parameters()['model/vf_fc1/w:0'])
    globals().update(locals())


    return 



if __name__ == '__main__':
    demo_saving_and_loading_of_imitation_network()
































"""
# This is a print out of the observation and action from running a motion imitation test
# I think the spaces have been "flattened" so perhaps it's not a good way to determine the shape

OMGOMGOMG observation, action
 [-4.78984065e-01  5.34568716e-02 -1.63287449e+00  5.32900000e+00
 -4.36243862e-01 -6.62830272e-02 -1.84114265e+00  3.65008640e+00
 -3.50700434e-01 -9.81579764e-02 -2.82752442e+00  4.57750738e-01
 -7.51813889e-01  3.62653686e-01 -1.13134554e+00  1.34970829e-01
  1.52431814e+00 -1.39375967e+00  4.93190825e-01  9.31204273e-01
 -1.41303985e+00 -1.36074632e-01  5.66170780e-01 -1.20268941e+00
 -8.37637186e-01  4.92933823e-01 -1.08289291e+00  1.32836387e-01
  1.69829158e+00 -1.41731989e+00  3.05400401e-01  9.02953951e-01
 -1.48772553e+00 -2.55377620e-01  5.36985470e-01 -1.39744517e+00
 -6.96970761e-01  4.28752510e-01 -1.01704553e+00  1.19780384e-01
  1.67330603e+00 -1.38047607e+00  3.17495912e-01  8.24308990e-01
 -1.62361851e+00 -2.49079883e-01  4.40227015e-01 -1.39240116e+00
 -7.48754011e-01  3.75057383e-01 -1.04867834e+00  1.03956754e-01
  1.65625349e+00 -1.42325394e+00  2.61049825e-01  9.08830032e-01
 -1.57991793e+00 -3.47652701e-01  4.69471095e-01 -1.58970455e+00
 -7.87208098e-01  3.67459892e-01 -1.05036642e+00  1.23221771e-01
  1.71240954e+00 -1.42636541e+00  3.03851231e-01  1.02821362e+00
 -1.65059533e+00 -2.90858958e-01  5.96602030e-01 -1.68725138e+00
 -9.64509368e-01  4.90061082e-01 -1.04339826e+00  7.96501471e-02
  1.92113864e+00 -1.36039782e+00  3.41579558e-01  1.05391173e+00
 -1.66501889e+00 -2.83522476e-01  5.34727059e-01 -1.59965214e+00
  3.29540703e-02  1.18993801e-02  1.89451248e-03  3.83527202e-01
  6.05337231e-01  6.04850042e-01  3.47318512e-01 -3.41633628e-02
 -4.32393972e-01 -7.18639933e-01 -1.23663163e-01  2.80251502e-01
 -6.54077554e-01 -1.96210220e-01 -2.96891331e-01 -6.19670041e-01
 -1.86565333e-01  3.97121581e-01 -6.35660176e-01  6.76831112e-02
  2.63891511e-02 -2.09764630e-03  3.84632283e-01  6.01716527e-01
  6.08613665e-01  3.45810113e-01 -7.19901467e-02 -3.09718556e-01
 -7.70475470e-01 -1.22796027e-01  3.32451113e-01 -5.95395980e-01
 -2.00256364e-01 -1.85826757e-01 -6.93003114e-01 -1.78278899e-01
  4.98950401e-01 -7.48863234e-01  3.11443637e-01  1.63714069e-01
 -2.15263753e-03  3.22423076e-01  6.28963749e-01  6.11214741e-01
  3.56180437e-01 -1.42894038e-01  3.04964940e-01 -7.15399238e-01
 -2.14108239e-01 -5.28091887e-01 -7.70310261e-01 -2.75519886e-01
  4.35129730e-01 -6.46914429e-01 -1.06306823e-01 -3.76240818e-01
 -6.37947287e-01  9.47512674e-01  4.90550936e-01 -3.52453844e-03
  3.20588331e-01  6.28686897e-01  6.09827634e-01  3.60674596e-01
 -1.48648591e-01  3.43871416e-01 -6.75551227e-01 -1.53479468e-01
 -4.78108227e-01 -7.19452724e-01 -2.78910039e-01  5.04155164e-01
 -6.78555777e-01 -1.05447454e-01 -2.72583515e-01 -7.11850977e-01] 


 [-0.7098206  -0.27678287  0.14137934  0.2712906   0.8873661  -0.3362867
  0.6141649  -0.05463757 -0.10541792 -0.05253304 -0.17279938  0.16166855]
"""


"""
import gym
import numpy as np

gym.spaces.Box( low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32 )
gym.spaces.Box(2,)
"""


"""
print(model.get_parameters()['model/pi_fc0/w:0'].shape)
print(model.get_parameters()['model/pi_fc0/b:0'].shape)
print(model.get_parameters()['model/vf_fc0/w:0'].shape)
print(model.get_parameters()['model/vf_fc0/b:0'].shape)

>>> print(model2.get_parameters()['model/pi_fc0/w:0'].shape)
(28, 512)
>>> print(model2.get_parameters()['model/pi_fc0/b:0'].shape)
(512,)
>>> print(model2.get_parameters()['model/vf_fc0/w:0'].shape)
(28, 512)
>>> print(model2.get_parameters()['model/vf_fc0/b:0'].shape)
(512,)


>>> print(model.get_parameters()['model/pi_fc0/w:0'].shape)
(160, 512)
>>> print(model.get_parameters()['model/pi_fc0/b:0'].shape)
(512,)
>>> print(model.get_parameters()['model/vf_fc0/w:0'].shape)
(160, 512)
>>> print(model.get_parameters()['model/vf_fc0/b:0'].shape)
(512,)


"""




# def save_using_tensorflow():
#       pass
#       # def __init__(self):
#       #        vars_with_adam = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"/")
#       #        self.vars = [v for v in vars_with_adam if 'Adam' not in v.name]
#       #        self.policy_saver = tf.train.Saver(var_list=self.vars)
#       #        self.t1 = time.time()

#       # def save(self):
#       #        self.policy_saver.save(tf.get_default_session(), self.PATH + 'model.ckpt')

#       # def load(self, WEIGHT_PATH):
#       #        self.policy_saver.restore(tf.get_default_session(), WEIGHT_PATH + 'model.ckpt')
#       #        print("Loaded weights for " + self.name + " module")

