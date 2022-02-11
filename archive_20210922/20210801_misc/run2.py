# The action space and the observation space from the motion imitation system
# print(env.action_space, env.observation_space)
# >> WOOO Box(12,) Dict(IMU:Box(4,), LastAction:Box(12,), MotorAngle:Box(12,))

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')
import numpy as np
import argparse
import tensorboardX
from collections import deque   
from scripts.utils import *
from scripts.mpi_utils import sync_from_root
import random
from mpi4py import MPI
import time
from pathlib import Path
home = str(Path.home())
from scripts import logger
import json
import git
np.set_printoptions(precision=3, suppress=True)
from dotmap import DotMap


def env_get_im(im_size=[48,48,4]):
    return np.zeros(im_size)


if __name__ == '__main__':

    # don't use a GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    # arguments
    args = DotMap(cur_decay='exp', decay_rate=1, decay=0.65, cur_local=True, cur_len=1200, cur_num=3, cur=False, stair_thing=True, obstacle_type='flat', show_detection=False, num_artifacts=2, 
    height_coeff=0.07, difficulty=1, detection_dist=0.9, more_power=1, MASTER=True, dist_off_ground=True, disturbances=True, record_step=True, dist_inc=0, initial_disturbance=100, 
    final_disturbance=100, dist_difficulty=0, expert=False, render=False, e2e=False, vis=True, vis_type='None', camera_rate=6, display_im=False, const_std=False, const_lr=False, max_ts=30000000.0, 
    lr=0.0003, vf_lr=0.0003, std_clip=False, separate_vf=False, lstm_pol=False, dual_value=False, dual_dqn=False, folder='hex', exp='test', control_type='walk', seed=42, eval=True, hpc=False, 
    test_pol=False, eval_first=False, sleep=0.01, dqn=False, debug=False, multi=False, all_setup=False, doa=False, adv=False, yu=False, nicks=False, rand_Kp=False, early_stop=True, inc=1, 
    terrain_first=True, advantage2=True, include_actions=False, single_pol=False, comparison=None, use_roa=False, baseline=False, rand_flat=False, new=False, box_pen=False, eval_dist=False, 
    vf_only=False, speed_cur=False, use_base=False, display_doa=False, act=False, forces=False, mocap=False, stage3=False, dqn_cur_decay=False, term=False, multi_robots=False, supervised=False, 
    min_eps=0.01, eps_decay=3000, min_decay=0.001, just_setup=False, just_dqn=False, old_rew=False, use_classifier=False, sim_type='pb', robot='hexapod', alg='ppo')

    args.folder   = '20200825'
    args.vis      = False
    args.test_pol = False


    PATH = home + '/results/hexapod/latest/' + args.folder + '/' + args.exp + '/'
    print("PATH", PATH)
    # multiprocessing stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # seed tensorflow's random number generator
    myseed = args.seed + 10000 * rank
    np.random.seed(myseed)
    random.seed(myseed)
    tf.set_random_seed(myseed)
    # logger    
    logger.configure(dir=PATH)
    if rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=PATH)
    else: 
        writer = None 
    # Set up a tensorflow session
    # sess = tf.Session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.1)
    sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                            intra_op_parallelism_threads=1,             
                                            gpu_options=gpu_options), graph=None)
    

    # Import the gym environment to simulate using the pybullet physics engine
    import laikago_env
    env = laikago_env.env_builder_build_laikago_env( motor_control_mode = laikago_env.robot_config.MotorControlMode.POSITION, enable_rendering=True)

    # Import the PPO neural network
    from models.ppo import Model
    horizon = 2048
    # model = Model(args.robot, env=env, ac_size=12, ob_size=28, args=args) # min args
    model = Model(args.robot, env=env, ob_size=28, ac_size=12, args=args, PATH=PATH, horizon=horizon, writer=writer, max_timesteps=args.max_ts, vis=args.vis)



    initialize()
    sync_from_root(sess, model.vars, comm=comm)
    model.set_training_params(max_timesteps=args.max_ts, learning_rate=args.lr, horizon=horizon)

    if args.test_pol:
        if args.hpc:
            model.load(home + '/hpc-home/results/hexapod/latest/' + args.folder + '/' + args.exp + '/')      
        else:
            model.load('./weights/hex/')      

    # Throw an error if the graph grows (shouldn't change once everything is initialised)
    tf.get_default_graph().finalize()
    
    prev_done = True
    # ===============================================================
    # You could change the curriculum parameters each episode by passing a list of cur_params
    # 1. Guide forces: variable = env.Kp (range = 400 - 0)
    # 2. Terrain difficulty: variable = env.difficulty (range = 1-10)
    # 3. Perturbations: variable = env.max_disturbance (range = 50 - 2000)
    # ===============================================================
    ob = env.reset()
    im = env_get_im()
    ep_ret = 0
    ep_len = 0
    ep_rets = []
    ep_lens = []
    ep_steps = 0

    if args.test_pol:
        stochastic = False
    else:
        stochastic = True


    # Main Loop
    while True: 
        if model.timesteps_so_far > model.max_timesteps:
            break 
        act, vpred, _, nlogp = model.step(ob, im, stochastic=stochastic)

        torques = act

        next_ob, rew, done, _ = env.step(torques)
        next_im = env_get_im()

        if not args.test_pol:
            model.add_to_buffer([ob, im, act, rew, prev_done, vpred, nlogp])
        prev_done = done
        ob = next_ob
        im = next_im
        ep_ret += rew
        ep_len += 1
        ep_steps += 1
        
        if not args.test_pol and ep_steps % horizon == 0:
            _, vpred, _, _ = model.step(next_ob, next_im, stochastic=True)
            model.run_train(data={"ep_rets":ep_rets, "ep_lens":ep_lens}, last_value=vpred, last_done=done)
            ep_rets = []
            ep_lens = []
        
        if done:
            # ===============================================================
            # You could change the curriculum parameters each episode by passing a list of cur_params
            # 1. Guide forces: variable = env.Kp (range = 400 - 0)
            # 2. Terrain difficulty: variable = env.difficulty (range = 1-10)
            # 3. Perturbations: variable = env.max_disturbance (range = 50 - 2000)
            # ===============================================================
            ob = env.reset(cur_params=None)
            im = env_get_im()
            ep_rets.append(ep_ret)  
            ep_lens.append(ep_len)     
            ep_ret = 0
            ep_len = 0      







# What's being passed into the predict function
# --------------------------------------------------------
# print("\n\nob[None], im[None]\n\n", ob[None], im[None])
# print("\n\nactions, values, self.states, neglogpacs\n\n", actions, values, self.states, neglogpacs)


#  [[  0.332  -0.056   1.534  -0.46    0.083   2.113  -0.077  -0.014   2.307
#     0.035  -0.062   2.21   -0.446  -0.212   1.699   0.342  -0.058   1.919
#    -2.581  -4.096 -10.246  -3.667  11.411  -8.994  -1.946  -2.726  39.68
#     1.064   2.693  16.177  -1.109 -15.732  30.52   -1.464  -4.445  23.636
#    -0.014  -0.341  -0.003   0.013  -0.012  -0.973   1.556   0.501   1.
#     1.      1.      1.      1.      1.      1.      1.      1.      1.
#     1.      1.   ]] 

# [[[[0.]
#    [0.]
#    [0.]
#    ...
#    [0.]
#    [0.]
#    [0.]]]]


# actions, values, self.states, neglogpacs

#  [[-0.596  0.736 -0.111 -0.954 -0.329 -0.233  0.393 -1.197  1.355 -0.707
#   -0.303 -0.521 -0.251  0.333  1.312  1.098 -0.001 -1.028]] -0.40250462 None 21.794088


