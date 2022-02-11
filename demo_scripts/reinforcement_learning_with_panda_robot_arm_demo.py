
# import tensorflow as tf
import stable_baselines as sb
import gym
import pybullet as pb
import pybullet_data
import numpy as np

import os, math, random, datetime

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import PPO1
# import panda_env
# from gym import error, spaces, utils
# from gym.utils import seeding

VERBOSE = False


class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pb.connect(pb.GUI)
        pb.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = gym.spaces.Box(np.array([-math.pi]*12), np.array([math.pi]*12))
        self.observation_space = gym.spaces.Box(np.array([-math.pi]*12), np.array([math.pi]*12))

    def reset(self):
        pb.resetSimulation()
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        pb.setGravity(0,0,-10)
        urdfRootPath=pybullet_data.getDataPath()
        

        planeUid = pb.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0]*12 # [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid = pb.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            pb.resetJointState(self.pandaUid,i, rest_poses[i])

        tableUid = pb.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        trayUid = pb.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = pb.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)

        joint_states = pb.getJointStates(self.pandaUid, [0,1,2,3,4,5,6,7,8,9,10,11] )
        joint_states = [ joint_state[0] for joint_state in joint_states ]
        # print(joint_states)
        observation = np.asarray(joint_states)

        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,1) # rendering's back on again
        return observation


    def step(self, action):
        # print(action)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)

        pb.setJointMotorControlArray(self.pandaUid, [0,1,2,3,4,5,6,7,8,9,10,11], pb.POSITION_CONTROL, action)
        pb.stepSimulation()

        link_state = pb.getLinkState( self.pandaUid, 10, computeForwardKinematics = True )
        panda_position, panda_orientation = link_state[0], link_state[1]
        object_position, object_orientation = pb.getBasePositionAndOrientation(self.objectUid)
        distance_between_panda_fingers_and_object = np.sqrt(np.sum(np.square(np.array(panda_position)-np.array(object_position))))
        # print(distance_between_panda_fingers_and_object)
        reward = 1-distance_between_panda_fingers_and_object
        if VERBOSE: print(reward)

        joint_states = pb.getJointStates(self.pandaUid, [0,1,2,3,4,5,6,7,8,9,10,11] )
        joint_states = [ joint_state[0] for joint_state in joint_states ]
        if VERBOSE: print(joint_states)
        observation = np.asarray(joint_states)
        # time.sleep(0.1)
        done, info = False, False 
        return observation, reward, done, info


    def render(self, mode='human'):

        view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.7,0,0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(960) /720,
            nearVal=0.1,
            farVal=100.0
        )

        (_, _, px, _, _) = pb.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        pb.disconnect()





def main():

    env = PandaEnv()
    model = sb.PPO1(sb.common.policies.MlpPolicy, env, verbose=1)

    observation = env.reset()
    
    # # train
    # for i in range(1000):
    #     model.learn(total_timesteps=1000)
    #     filename = 'panda_ppo_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + str(i)
    #     model.save(filename)


    # test
    model.load_parameters("panda_ppo_2")
    while True:
        t = 0
        observation = env.reset()
        print("RESET")
        while t < 100:
            t += 1
            action, _ = model.predict(observation, deterministic=True)
            observation, r, done, info = env.step(action)
            # if done:
            #     observation = env.reset()



if __name__ == '__main__':
    main()







# # max_timesteps=int(1e6), lr=3e-4, horizon=2048, batch_size=32


# # hexapod PPO args
# # def __init__(self, name, env, ac_size, ob_size, im_size=[48,48,4], args=None, PATH=None, writer=None, hid_size=256, vis=False, normalize=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, mpi_rank_weight=1, max_timesteps=int(1e6), lr=3e-4, horizon=2048, batch_size=32, const_std=False):

# # # default arguments
# # args = DotMap(cur_decay='exp', decay_rate=1, decay=0.65, cur_local=True, cur_len=1200, cur_num=3, cur=False, stair_thing=True, obstacle_type='flat', show_detection=False, num_artifacts=2, 
# # height_coeff=0.07, difficulty=1, detection_dist=0.9, more_power=1, MASTER=True, dist_off_ground=True, disturbances=True, record_step=True, dist_inc=0, initial_disturbance=100, 
# # final_disturbance=100, dist_difficulty=0, expert=False, render=False, e2e=False, vis=True, vis_type='None', camera_rate=6, display_im=False, const_std=False, const_lr=False, max_ts=30000000.0, 
# # lr=0.0003, vf_lr=0.0003, std_clip=False, separate_vf=False, lstm_pol=False, dual_value=False, dual_dqn=False, folder='hex', exp='test', control_type='walk', seed=42, eval=True, hpc=False, 
# # test_pol=False, eval_first=False, sleep=0.01, dqn=False, debug=False, multi=False, all_setup=False, doa=False, adv=False, yu=False, nicks=False, rand_Kp=False, early_stop=True, inc=1, 
# # terrain_first=True, advantage2=True, include_actions=False, single_pol=False, comparison=None, use_roa=False, baseline=False, rand_flat=False, new=False, box_pen=False, eval_dist=False, 
# # vf_only=False, speed_cur=False, use_base=False, display_doa=False, act=False, forces=False, mocap=False, stage3=False, dqn_cur_decay=False, term=False, multi_robots=False, supervised=False, 
# # min_eps=0.01, eps_decay=3000, min_decay=0.001, just_setup=False, just_dqn=False, old_rew=False, use_classifier=False, sim_type='pb', robot='hexapod', alg='ppo')


# # const learning rate is false
# # max timesteps = 300e6 ?
# # lr=0.0003, vf_lr=0.0003, std_clip=False, separate_vf=False, lstm_pol=False, 
# # control typet = walk
# # seed = 42
# # # min_eps=0.01, eps_decay=3000, min_decay=0.001
# # entcoeff = 0.01
# # const_lr=False




# # gym PPO args
# #     def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01, optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5, schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):



# # the ranges of my motors don't seem to be getting clipped properly.  the range of motion is too large
# # entropy coefficient is zero in hexapod PPO
# # learning rate is the same
# # learning rate seems to change over time (linearly?)





# # reward functions
# # ----------------------
# # 
# # just stand still for a long time.
# # change orientation of body - move head around
# # go forward 1 meter
# # go backward 1 meter
# # move in any direction
# # move forward
# # 
# # reward laziness
# # reward not moving
# # reward standing up and being stable

# # neural pruning
# # the leg controllers should be the same.  They should be the same but out of phase.
# # to the extent that they are the same it's good.



# # keep feet on the ground and then maximize stability of the body


# # points for forward or backwards
# #   def reward(self, env):
# #     """Get the reward without side effects."""
# #     del env
# #     return abs(self.current_base_pos[0] - self.last_base_pos[0])
