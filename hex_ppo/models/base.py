'''
Base class for all networks.
For now includes training parameters..
'''
# import scripts.utils as U
import tensorflow as tf
from mpi4py import MPI
import time
import numpy as np
from collections import deque
from hex_ppo.scripts import logger
from hex_ppo.scripts.distributions import make_pdtype

class Base():
    epochs = 10
    batch_size = 32
    horizon = 2048
    enable_shuffle = True
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    best_reward = 0
    tstart = time.time()
    t1 = time.time()
    all_rewards = []
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    def __init__(self):
        vars_with_adam = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"/")
        self.vars = [v for v in vars_with_adam if 'Adam' not in v.name]
        self.policy_saver = tf.train.Saver(var_list=self.vars)
        self.t1 = time.time()

    def save(self):
        self.policy_saver.save(tf.get_default_session(), self.PATH + 'model.ckpt')

    def load(self, WEIGHT_PATH):
        self.policy_saver.restore(tf.get_default_session(), WEIGHT_PATH + 'model.ckpt')
        print("Loaded weights for " + self.name + " module")

    def evaluate(self, rews, lens):

        lrlocal = (rews, lens) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        rews, lens = map(flatten_lists, zip(*listoflrpairs))
        self.lenbuffer.extend(lens)
        self.rewbuffer.extend(rews)
        if self.rank == 0:
            logger.log("********** Iteration %i ************"%self.iters_so_far)
            for loss, name in zip(self.loss, self.loss_names):
                logger.record_tabular(name, loss)
                self.writer.add_scalar(name, loss, self.iters_so_far)
            logger.record_tabular("LearningRate", self.lr)
            logger.record_tabular("EpRewMean", np.mean(self.rewbuffer))
            logger.record_tabular("EpLenMean", np.mean(self.lenbuffer))
            logger.record_tabular("EpThisIter", len(self.lenbuffer))
            logger.record_tabular("EpThisIter", len(self.lenbuffer))
            logger.record_tabular("TimeThisIter", time.time() - self.t1)
            logger.record_tabular("TimeStepsSoFar", self.timesteps_so_far)
            # logger.record_tabular("EnvTotalSteps", self.env.total_steps)
            self.t1 = time.time()
            self.writer.add_scalar("rewards", np.mean(self.rewbuffer), self.iters_so_far)
            self.writer.add_scalar("lengths", np.mean(self.lenbuffer), self.iters_so_far)
            self.all_rewards.append(np.mean(self.rewbuffer))
        
        # self.env.log_stuff(logger, self.writer, self.iters_so_far)
        
        if self.rank == 0:
            np.save(self.PATH + 'rewards.npy',self.all_rewards)
            try:
                # if np.mean(self.rewbuffer) > self.best_reward :
                self.save()
            except:
                print("Couldn't save training model")
            logger.dump_tabular()
        
        
        self.timesteps_so_far += sum(lens)     
        self.iters_so_far += 1


    def set_training_params(self, max_timesteps, learning_rate, horizon):
        self.max_timesteps = max_timesteps
        self.learning_rate = learning_rate
        self.horizon = horizon

    def log_stuff(self):
        pass

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]