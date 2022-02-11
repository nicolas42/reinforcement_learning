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

from motion_imitation.robots import robot_config
from motion_imitation.envs import env_builder as env_builder
from models.ppo import Model
from motion_imitation.robots import laikago

def doit():

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
    
    horizon = 2048





    
    env = env_builder.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True)
    # pol = Model(args.robot, env=env, ac_size=12, ob_size=28, args=args) # min args
    pol = Model(args.robot, env=env, ob_size=28, ac_size=12, args=args, PATH=PATH, horizon=horizon, writer=writer, max_timesteps=args.max_ts, vis=args.vis)




    # observation = env.reset()
    # while 1:

    #     action = laikago.INIT_MOTOR_ANGLES
    #     # action =  [ 0.,    0.67, -1.25,  0.,    0.67, -1.25,  0.,    0.67, -1.25,  0.,    0.67, -0] 
    #     print(action)
    #     # The action is essentially the target position
    #     # to use a model. pass it the current state (observation) and it will predict an action, (and a modified state?)

    #     # action, _ = model.predict(observation, deterministic=True)
    #     observation, reward, done, info = env.step(action)

    #     print("observation,reward,done,action\n", observation,"\n", reward,"\n", done,"\n", action,"\n")
    #     # time.sleep(0.1)
    #     if done:
    #         observation = env.reset()
    #         print("DONE")
    #         break







    # from assets.env_pb_hex import Env
    # env = Env(PATH=PATH, args=args)
    # from models.ppo import Model
    # pol = Model(args.robot, env=env, ob_size=env.ob_size, ac_size=env.ac_size, im_size=env.im_size, args=args, PATH=PATH, horizon=horizon, writer=writer, max_timesteps=args.max_ts, vis=args.vis)



    initialize()
    sync_from_root(sess, pol.vars, comm=comm)
    pol.set_training_params(max_timesteps=args.max_ts, learning_rate=args.lr, horizon=horizon)

    if args.test_pol:
        if args.hpc:
            pol.load(home + '/hpc-home/results/hexapod/latest/' + args.folder + '/' + args.exp + '/')      
        else:
            pol.load('./weights/hex/')      

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
    im = np.zeros([48,48,4])
    ep_ret = 0
    ep_len = 0
    ep_rets = []
    ep_lens = []
    ep_steps = 0

    if args.test_pol:
        stochastic = False
    else:
        stochastic = True
    
    # stochastic = False ### stop it going crazy

    while True: 
        if pol.timesteps_so_far > pol.max_timesteps:
            break 
        act, vpred, _, nlogp = pol.step(ob, im, stochastic=stochastic)
        print(act)
        # act = laikago.INIT_MOTOR_ANGLES # [ 0.    0.67 -1.25  0.    0.67 -1.25  0.    0.67 -1.25  0.    0.67 -1.25]

        next_ob, rew, done, _ = env.step(act)
        next_im = np.zeros([48,48,4])

        if not args.test_pol:
            pol.add_to_buffer([ob, im, act, rew, prev_done, vpred, nlogp])
        prev_done = done
        ob = next_ob
        im = next_im
        ep_ret += rew
        ep_len += 1
        ep_steps += 1
        
        if not args.test_pol and ep_steps % horizon == 0:
            _, vpred, _, _ = pol.step(next_ob, next_im, stochastic=True)
            pol.run_train(data={"ep_rets":ep_rets, "ep_lens":ep_lens}, last_value=vpred, last_done=done)
            ep_rets = []
            ep_lens = []
        
        if done:
            # ===============================================================
            # You could change the curriculum parameters each episode by passing a list of cur_params
            # 1. Guide forces: variable = env.Kp (range = 400 - 0)
            # 2. Terrain difficulty: variable = env.difficulty (range = 1-10)
            # 3. Perturbations: variable = env.max_disturbance (range = 50 - 2000)
            # ===============================================================
            ob = env.reset()
            im = np.zeros([48,48,4])
            ep_rets.append(ep_ret)  
            ep_lens.append(ep_len)     
            ep_ret = 0
            ep_len = 0        



if __name__ == '__main__':
    doit()