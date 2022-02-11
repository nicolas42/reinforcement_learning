import os, inspect
os.sys.path.insert(0, "hex_ppo") # os.path.abspath("hex_ppo"))

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

def run(args):

    PATH = home + '/results/hexapod/latest/' + args.folder + '/' + args.exp + '/'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    myseed = args.seed + 10000 * rank
    np.random.seed(myseed)
    random.seed(myseed)
    tf.set_random_seed(myseed)
    
    logger.configure(dir=PATH)
    if rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=PATH)
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        with open(PATH + 'commandline_args.txt', 'w') as f:
            f.write('Hash:')
            f.write(str(sha) + "\n")
            json.dump(args.__dict__, f, indent=2)
    else: 
            writer = None 

    # sess = tf.Session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.1)
    sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                            intra_op_parallelism_threads=1,             
                                            gpu_options=gpu_options), graph=None)
    
    horizon = 2048

    from assets.env_pb_hex import Env
    
    env = Env(PATH=PATH, args=args)

    from models.ppo import Model

    pol = Model(args.robot, env=env, ob_size=env.ob_size, ac_size=env.ac_size, im_size=env.im_size, args=args, PATH=PATH, horizon=horizon, writer=writer, max_timesteps=args.max_ts, vis=args.vis)
    
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
    ob = env.reset(cur_params=None)
    im = env.get_im()
    ep_ret = 0
    ep_len = 0
    ep_rets = []
    ep_lens = []
    ep_steps = 0

    if args.test_pol:
        stochastic = False
    else:
        stochastic = True
    
    while True: 
        if pol.timesteps_so_far > pol.max_timesteps:
            break 
        act, vpred, _, nlogp = pol.step(ob, im, stochastic=stochastic)

        # print(act)
        torques = act

        next_ob, rew, done, _ = env.step(torques)
        next_im = env.get_im()

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
            ob = env.reset(cur_params=None)
            im = env.get_im()
            ep_rets.append(ep_ret)  
            ep_lens.append(ep_len)     
            ep_ret = 0
            ep_len = 0        

if __name__ == '__main__':
    # Hack for loading defaults, but still accepting run specific defaults
    import defaults
    from dotmap import DotMap
    args1, unknown1 = defaults.get_defaults() 
    parser = argparse.ArgumentParser()
    # # default arguments
    # args = DotMap(cur_decay='exp', decay_rate=1, decay=0.65, cur_local=True, cur_len=1200, cur_num=3, cur=False, stair_thing=True, obstacle_type='flat', show_detection=False, num_artifacts=2, 
    # height_coeff=0.07, difficulty=1, detection_dist=0.9, more_power=1, MASTER=True, dist_off_ground=True, disturbances=True, record_step=True, dist_inc=0, initial_disturbance=100, 
    # final_disturbance=100, dist_difficulty=0, expert=False, render=False, e2e=False, vis=True, vis_type='None', camera_rate=6, display_im=False, const_std=False, const_lr=False, max_ts=30000000.0, 
    # lr=0.0003, vf_lr=0.0003, std_clip=False, separate_vf=False, lstm_pol=False, dual_value=False, dual_dqn=False, folder='hex', exp='test', control_type='walk', seed=42, eval=True, hpc=False, 
    # test_pol=False, eval_first=False, sleep=0.01, dqn=False, debug=False, multi=False, all_setup=False, doa=False, adv=False, yu=False, nicks=False, rand_Kp=False, early_stop=True, inc=1, 
    # terrain_first=True, advantage2=True, include_actions=False, single_pol=False, comparison=None, use_roa=False, baseline=False, rand_flat=False, new=False, box_pen=False, eval_dist=False, 
    # vf_only=False, speed_cur=False, use_base=False, display_doa=False, act=False, forces=False, mocap=False, stage3=False, dqn_cur_decay=False, term=False, multi_robots=False, supervised=False, 
    # min_eps=0.01, eps_decay=3000, min_decay=0.001, just_setup=False, just_dqn=False, old_rew=False, use_classifier=False, sim_type='pb', robot='hexapod', alg='ppo')


    # Arguments that are specific for this run (including run specific defaults, ignore unknown arguments)
    parser.add_argument('--folder', default='hex')
    parser.add_argument('--exp', default='test')
    parser.add_argument('--difficulty', default=1, type=int)
    parser.add_argument('--obstacle_type', default='flat', help='Obstacle types are: flat, gaps, jumps, stairs, steps, high_jumps, one_leg_hop, hard_steps, hard_high_jumps, or mix (for a combination)')
    parser.add_argument('--sim_type', default='pb', help='pb or gz (pybullet or gazebo)')
    parser.add_argument('--robot', default='hexapod', help='biped or hexapod')
    parser.add_argument('--alg', default='ppo', help='ppo') 
    parser.add_argument('--max_ts', default=3e7, type=int)
    parser.add_argument('--vis_type', default='None') 
    args2, unknown2 = parser.parse_known_args()
    args2 = vars(args2)
    # Replace any arguments from defaults with run specific defaults
    for key in args2:
        args1[key] = args2[key]
    # Look for any changes to defaults (unknowns) and replace values in args1
    for n, unknown in enumerate(unknown2):
        if "--" in unknown and n < len(unknown2)-1 and "--" not in unknown2[n+1]:
            arg_type = type(args1[unknown[2:]])
            args1[unknown[2:]] = arg_type(unknown2[n+1])
    args = DotMap(args1)
    # Check for dodgy arguments
    unknowns = []
    for unknown in unknown1 + unknown2:
        if "--" in unknown and unknown[2:] not in args:
            unknowns.append(unknown)
    if len(unknowns) > 0:
        print("Dodgy argument")
        print(unknowns)
        exit()

    args.render=True
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    run(args)
