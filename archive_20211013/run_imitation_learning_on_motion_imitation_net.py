import numpy as np
import tensorflow as tf
import stable_baselines as sb
import os, sys
os.sys.path.append(".")
# from mpi4py import MPI

from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation

from motion_imitation.robots import robot_config
# from motion_imitation.envs import env_builder as env_builder
import env_builder as env_builder

global_policy_kwargs = {
    "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
    "act_fun": tf.nn.relu
}

def load_imitation_model(args_imitation_ppo_filename):
    imitation_environment = env_builder.build_imitation_env( 
            motion_files=["motion_imitation/data/motions/dog_pace.txt"], num_parallel_envs=1, mode="test", enable_randomizer=True, enable_rendering=False)
    # from run0_imitation_learning import build_model 
    # imitation_model = build_model(imitation_environment, num_procs = 1 , timesteps_per_actorbatch = 4096, optim_batchsize = 256, output_dir = "output" )
    imitation_model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=imitation_environment,
        gamma=0.95,
        timesteps_per_actorbatch=4096,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=256,
        lam=0.95,
        adam_epsilon=1e-5,
        schedule='constant',
        policy_kwargs=global_policy_kwargs,
        tensorboard_log="output",
        verbose=1
    )
    imitation_model.load_parameters(args_imitation_ppo_filename)
    imitation_environment._pybullet_client.__del__() # pybullet.disconnect
    return imitation_model


def copy_imitation_parameters_to_new_model(imitation_model, new_model):

    imitation_params = imitation_model.get_parameters()
    # imitation_params['model/pi_fc0/w:0'] = imitation_params['model/pi_fc0/w:0'][0:28]
    # imitation_params['model/vf_fc0/w:0'] = imitation_params['model/pi_fc0/w:0'][0:28]

    offset = 0
    imitation_params['model/pi_fc0/w:0'] = np.concatenate( (
        imitation_params['model/pi_fc0/w:0'][offset+0:offset+12],
        imitation_params['model/pi_fc0/w:0'][offset+36:offset+40],
        imitation_params['model/pi_fc0/w:0'][offset+48:offset+60]
    ))

    imitation_params['model/vf_fc0/w:0'] = np.concatenate( (
        imitation_params['model/vf_fc0/w:0'][offset+0:offset+12],
        imitation_params['model/vf_fc0/w:0'][offset+36:offset+40],
        imitation_params['model/vf_fc0/w:0'][offset+48:offset+60]
    ))

    new_model.load_parameters(imitation_params, exact_match=False) 
    # the imitation model doesn't have a logstd layer so exact_match=False is needed
    # the logstd layer is not updated.
    return new_model


def run():


    imitation_model = load_imitation_model("motion_imitation/data/policies/dog_pace.zip")

    env = env_builder.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True )

    model = sb.PPO1(sb.common.policies.MlpPolicy, env, verbose=1, policy_kwargs=global_policy_kwargs )


    copy_imitation_parameters_to_new_model(imitation_model, model)


    # train
    for i in range(1000):
        model.learn(total_timesteps=10000)
        model.save("my_ppo_" + str(i))


    # test
    observation = env.reset()
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, r, done, info = env.step(action)
        if done:
            observation = env.reset()

if __name__ == '__main__':
    run()







# max_timesteps=int(1e6), lr=3e-4, horizon=2048, batch_size=32


# hexapod PPO args
# def __init__(self, name, env, ac_size, ob_size, im_size=[48,48,4], args=None, PATH=None, writer=None, hid_size=256, vis=False, normalize=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, mpi_rank_weight=1, max_timesteps=int(1e6), lr=3e-4, horizon=2048, batch_size=32, const_std=False):

# # default arguments
# args = DotMap(cur_decay='exp', decay_rate=1, decay=0.65, cur_local=True, cur_len=1200, cur_num=3, cur=False, stair_thing=True, obstacle_type='flat', show_detection=False, num_artifacts=2, 
# height_coeff=0.07, difficulty=1, detection_dist=0.9, more_power=1, MASTER=True, dist_off_ground=True, disturbances=True, record_step=True, dist_inc=0, initial_disturbance=100, 
# final_disturbance=100, dist_difficulty=0, expert=False, render=False, e2e=False, vis=True, vis_type='None', camera_rate=6, display_im=False, const_std=False, const_lr=False, max_ts=30000000.0, 
# lr=0.0003, vf_lr=0.0003, std_clip=False, separate_vf=False, lstm_pol=False, dual_value=False, dual_dqn=False, folder='hex', exp='test', control_type='walk', seed=42, eval=True, hpc=False, 
# test_pol=False, eval_first=False, sleep=0.01, dqn=False, debug=False, multi=False, all_setup=False, doa=False, adv=False, yu=False, nicks=False, rand_Kp=False, early_stop=True, inc=1, 
# terrain_first=True, advantage2=True, include_actions=False, single_pol=False, comparison=None, use_roa=False, baseline=False, rand_flat=False, new=False, box_pen=False, eval_dist=False, 
# vf_only=False, speed_cur=False, use_base=False, display_doa=False, act=False, forces=False, mocap=False, stage3=False, dqn_cur_decay=False, term=False, multi_robots=False, supervised=False, 
# min_eps=0.01, eps_decay=3000, min_decay=0.001, just_setup=False, just_dqn=False, old_rew=False, use_classifier=False, sim_type='pb', robot='hexapod', alg='ppo')


# const learning rate is false
# max timesteps = 300e6 ?
# lr=0.0003, vf_lr=0.0003, std_clip=False, separate_vf=False, lstm_pol=False, 
# control typet = walk
# seed = 42
# # min_eps=0.01, eps_decay=3000, min_decay=0.001
# entcoeff = 0.01
# const_lr=False




# gym PPO args
#     def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01, optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5, schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):



# the ranges of my motors don't seem to be getting clipped properly.  the range of motion is too large
# entropy coefficient is zero in hexapod PPO
# learning rate is the same
# learning rate seems to change over time (linearly?)





# reward functions
# ----------------------
# 
# just stand still for a long time.
# change orientation of body - move head around
# go forward 1 meter
# go backward 1 meter
# move in any direction
# move forward
# 
# reward laziness
# reward not moving
# reward standing up and being stable

# neural pruning
# the leg controllers should be the same.  They should be the same but out of phase.
# to the extent that they are the same it's good.



# keep feet on the ground and then maximize stability of the body


# points for forward or backwards
#   def reward(self, env):
#     """Get the reward without side effects."""
#     del env
#     return abs(self.current_base_pos[0] - self.last_base_pos[0])
