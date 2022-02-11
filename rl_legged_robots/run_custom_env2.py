###
# This is a modified version of openai-gym's run.py script created for running a custom environment
###

import sys
import gym
import os
import time
import random
import numpy as np
from os import path
import argparse as ap
from baselines.ppo2.model import Model
from collections import deque
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from baselines.run import get_alg_module, get_learn_function, get_learn_function_defaults
from baselines.common.policies import build_policy
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2.runner import Runner
from baselines.common import explained_variance, set_global_seeds
import tensorflow as tf
from baselines.ppo2.ppo2 import safemean
import os.path as osp

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

def parse_args():
    parser = ap.ArgumentParser()

    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--algorithm', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--cliprange', default=0.2, type=float)
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--noptepochs', type=int, default=4)
    parser.add_argument('--nminibatches', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--total_timesteps', type=int, default=1e6)
    parser.add_argument('--nsteps', type=int, default=100)
    known_args, unknown_args = parser.parse_known_args()

    unknown_arg_vals = {}
    arg_val = False
    for arg in unknown_args:
        if arg.startswith('--'):
            if '=' in arg:
                split_up = arg.split('=')
                key = split_up[0][2:]
                val = split_up[1]
                unknown_arg_vals[key] = val
            else:
                key = arg[2:]
                arg_val = True
        elif arg_val:
            unknown_arg_vals[key] = arg
            arg_val = False

    for key, val in unknown_arg_vals.items():
        try:
            new_val = eval(val)
        except:
            new_val = val
        unknown_arg_vals[key] = new_val

    return known_args, unknown_arg_vals


def prep_env(args):
    env_id = None
    for env in gym.envs.registry.all():
        if env.id == args.env:
            env_id = env.id
            env_type = env.entry_point.split(":")[0]
            env_type = env_type.split(".")[-1]
    assert env_id is not None, "Given openai gym environment not found."
    
    alg = args.algorithm
    num_envs = args.num_env or 1 #multiprocessing.cpu_count()
    seed = args.seed

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, num_envs, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)
    elif env_type in {'mujoco'}:
        print("MuJoCo not currently supported")
        exit(1)
    else:
        try:
            config = tf.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
            )
        except:
            config = tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
            )
        config.gpu_options.allow_growth = True
        get_session(config=config)
        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, num_envs, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

    return env_type, env_id, env


def make_callback(constant):
    def callback(_):
        return constant
    return callback

if __name__=="__main__":
    #parse command line args into known and unknown args
    known, unknown = parse_args()
    #create logger
    logger.configure()
    env_type, env_id, env = prep_env(known)

    #load the rl algorithm used
    learn_functn = get_learn_function(known.algorithm)

    #create arguments for that alg
    alg_kwargs = get_learn_function_defaults(known.algorithm, env_type)
    alg_kwargs.update(unknown)
    alg_kwargs['checkpoint_path'] = known.save_path

    policy = build_policy(env, known.network)
    nenvs = 1
    ac_space = env.action_space
    ob_space = env.observation_space

    nbatch = nenvs * known.nsteps
    nbatch_train = nbatch // known.nminibatches


    #some properties need to be callable by gym rather than constants, so make them callable if they're not
    if isinstance(known.learning_rate, float):
        lr = make_callback(known.learning_rate)
    if isinstance(known.cliprange, float):
        cliprange = make_callback(known.cliprange)

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                    nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=known.nsteps, ent_coef=known.ent_coef, vf_coef=known.vf_coef,
                    max_grad_norm=known.max_grad_norm)

    if known.load_path is not None:
        model.load(known.load_path)

    runner = Runner(env=env, model=model, nsteps=known.nsteps, gamma=known.gamma, lam=known.lam)
    epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    nupdates = int(known.total_timesteps//nbatch)
    for update in range(1, nupdates+1):
        assert nbatch % known.nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(known.noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % known.nminibatches == 0
            envsperbatch = nenvs // known.nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * known.nsteps).reshape(nenvs, known.nsteps)
            envsperbatch = nbatch_train // known.nsteps
            for _ in range(known.noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % known.log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*known.nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if known.save_interval and (update % known.save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
        
    env.render(True)
    env.reset()
    done = False
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    while not done:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, _, done, _ = env.step(actions)
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

    env.close()

