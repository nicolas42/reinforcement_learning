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

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import locomotion_gym_env
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.envs.env_wrappers import imitation_wrapper_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from motion_imitation.envs.env_wrappers import trajectory_generator_wrapper_env
from motion_imitation.envs.env_wrappers import simple_openloop
# from motion_imitation.envs.env_wrappers import simple_forward_task
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import default_task

from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import sensor_wrappers
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
# from motion_imitation.robots import laikago
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
import laikago

import numpy as np
import math 



def calculate_reward(self, env):
    uid = env.robot.quadruped
    pyb = env._pybullet_client

    base_velocity,_ = pyb.getBaseVelocity(uid)
    z = self.current_base_pos[2]

    # default_initial_orientation = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])
    position, orientation = pyb.getBasePositionAndOrientation(uid)
    orientation = pyb.getEulerFromQuaternion(orientation)
    # Euler angles
    yaw = orientation[2]
    yaw -= math.pi/2 
    if yaw < -math.pi:
      yaw += 2*math.pi 

    roll = orientation[0]
    roll -= math.pi/2 
    if roll < -math.pi:
      roll += 2*math.pi 

    joints = np.array(self.current_motor_angles)

    # Energy - avoid spasms
    # rewards small adjustments over large ones
    E = 10*(np.exp(-env.robot.GetEnergyConsumptionPerControlStep()))
  
    # Height - don't lie down. sigmoid, above 0.3m
    # easy
    H = 1/(1+np.exp(-100*(z-0.3)))

    # # Orientation - don't go sideways
    # # max reward = 3. goes to zero at +-60 degrees. Approx 1 at +-30 degrees.
    # O = 2*(np.exp(-5*yaw**2))
    # O = 0
    # # Orientation - roll
    # # O2 = (np.exp(-10*roll**2))
    # O2 = 0

    # # Velocity - go right.
    # V = (base_velocity[0]) 

    # P = -abs(position[1]) # don't go in y direction

    # print(base_velocity[0], V)
    # print({ "energy": round(E,1), "height": round(H,1), "roll": round(O2,1), "yaw": round(O,1), "velocity": round(V,1)})   # , "symmetry": round(S,1), "periodicity": round(P,1) })

    print(E,H) # ,V,P)
    reward = E + H # + V + P # O + V # + O2 # + S + P 
    del env
    return reward




def apply_external_force(env):
    # Apply external force
    # https://www.programcreek.com/python/example/122123/pybullet.applyExternalForce
    uid = env.robot.quadruped
    pyb = env._pybullet_client
    pyb.applyExternalForce(uid, -1, (1,1,1), (0, 0, 0), pyb.WORLD_FRAME)

def randomize_gravity(env):
    uid = env.robot.quadruped
    pyb = env._pybullet_client

    # Gravity Challenge :)
    # ---------------------------
    # pyb.setGravity(0,0,-9.8)
    # if (env._env_step_counter % 10000 == 0):
    g_x = np.random.normal(scale=1)
    g_y = np.random.normal(scale=1)
    g_z = -9.8 #  + np.random.normal(scale=3)
    print("\n\n\nCHANGE GRAVITY",g_x, g_y, g_z, "\n\n\n")
    pyb.setGravity(g_x,g_y,g_z)


class Reward(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.last_motor_angles = np.zeros(12)
    self.current_motor_angles = np.zeros(12)
    self.last_base_orientation = np.zeros(4)
    self.current_base_orientation = np.zeros(4)
    # self.joint_history = []


  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    # this occurs at the start of each episode
    # print("\n\n\nWOO\n\n\n")
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

    self.last_motor_angles = env.robot.GetMotorAngles()
    self.current_motor_angles = self.last_motor_angles

    
    self.last_base_orientation = env.robot.GetBaseOrientation()
    self.current_base_orientation = self.last_base_orientation

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()
    
    self.last_motor_angles = self.current_motor_angles
    self.current_motor_angles = env.robot.GetMotorAngles()

    self.last_base_orientation = self.current_base_orientation
    self.current_base_orientation = env.robot.GetBaseOrientation()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    # exit if robot is unstable
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    if rot_mat[-1] < 0.85:
      return 1
    
    return 0

  def reward(self, env):
    """Get the reward without side effects."""

    return calculate_reward(self,env)
    # return quaternion_reward(self,env)



def build_laikago_env( motor_control_mode, enable_rendering):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = True
  
  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = laikago.Laikago

  sensors = [
      robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS),
      robot_sensors.IMUSensor(),
      environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS)
  ]

  task = Reward()
  # from motion_imitation.envs.env_wrappers import simple_forward_task
  # task = simple_forward_task.SimpleForwardTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            robot_sensors=sensors, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND))

  return env


def build_imitation_env(motion_files, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering,
                        robot_class=laikago.Laikago,
                        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND)):
  assert len(motion_files) > 0

  curriculum_episode_length_start = 20
  curriculum_episode_length_end = 600
  
  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
  ]

  task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)

  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            env_randomizers=randomizers, robot_sensors=sensors, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                       trajectory_generator=trajectory_generator)

  if mode == "test":
      curriculum_episode_length_start = curriculum_episode_length_end

  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=30000000,
                                                  num_parallel_envs=num_parallel_envs)
  return env

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

    # # Attempt 1
    # offset = 0
    # imitation_params['model/pi_fc0/w:0'] = np.concatenate( (
    #     imitation_params['model/pi_fc0/w:0'][offset+0:offset+12],
    #     imitation_params['model/pi_fc0/w:0'][offset+36:offset+40],
    #     imitation_params['model/pi_fc0/w:0'][offset+48:offset+60]
    # ))

    # imitation_params['model/vf_fc0/w:0'] = np.concatenate( (
    #     imitation_params['model/vf_fc0/w:0'][offset+0:offset+12],
    #     imitation_params['model/vf_fc0/w:0'][offset+36:offset+40],
    #     imitation_params['model/vf_fc0/w:0'][offset+48:offset+60]
    # ))


    # It goes IMU, last action, motor angle, (alphabetical order)
    imitation_params['model/pi_fc0/w:0'] = np.concatenate( (
        imitation_params['model/pi_fc0/w:0'][0:4],
        imitation_params['model/pi_fc0/w:0'][12:24],
        imitation_params['model/pi_fc0/w:0'][48:60]
    ))
    print(imitation_params['model/pi_fc0/w:0'])

    imitation_params['model/vf_fc0/w:0'] = np.concatenate( (
        imitation_params['model/vf_fc0/w:0'][0:4],
        imitation_params['model/vf_fc0/w:0'][12:24],
        imitation_params['model/vf_fc0/w:0'][48:60]
    ))


    new_model.load_parameters(imitation_params, exact_match=False) 
    # the imitation model doesn't have a logstd layer so exact_match=False is needed
    # the logstd layer is not updated.
    return new_model



global_policy_kwargs = {
    "net_arch": [{"pi": [512, 256],"vf": [512, 256]}],
    "act_fun": tf.nn.relu
}


def demo(input_file):

    env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=True )

    model = sb.PPO1(sb.common.policies.MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                    optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear', 
                    policy_kwargs=global_policy_kwargs)


    model.load_parameters(input_file)

    observation = env.reset()
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, r, done, info = env.step(action)
        if done:
            observation = env.reset()



if __name__ == '__main__':

  if len(sys.argv) > 1 and sys.argv[1] == '--demo':
    input_file = sys.argv[2]
    demo(input_file)
  
  else:
    output_dir = "output/20211020_RL_on_motion_imitation_pretrained_net/" 
    os.mkdir(output_dir)

    imitation_model = load_imitation_model("motion_imitation/data/policies/dog_pace.zip")

    env = build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=False )

    model = sb.PPO1(sb.common.policies.MlpPolicy, env, verbose=1, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                    optim_stepsize=3e-4, optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear', 
                    tensorboard_log=output_dir + "tensorboard_log", policy_kwargs=global_policy_kwargs)


    copy_imitation_parameters_to_new_model(imitation_model, model)
    from stable_baselines.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(save_freq=2048, save_path=output_dir, name_prefix='' )
    model.learn(total_timesteps=2048000, callback=checkpoint_callback) # # , log_interval=100, tb_log_name="PPO1", reset_num_timesteps=True)







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




















# class MyTask(object):
#   """Default empy task."""
#   def __init__(self):
#     """Initializes the task."""
#     self.current_base_pos = np.zeros(3)
#     self.last_base_pos = np.zeros(3)
#     self.last_motor_angles = np.zeros(12)
#     self.current_motor_angles = np.zeros(12)
#     self.last_base_orientation = np.zeros(4)
#     self.current_base_orientation = np.zeros(4)

#   def __call__(self, env):
#     return self.reward(env)

#   def reset(self, env):
#     """Resets the internal state of the task."""
#     # print("woo")
#     self._env = env
#     self.last_base_pos = env.robot.GetBasePosition()
#     self.current_base_pos = self.last_base_pos

#     self.last_motor_angles = env.robot.GetMotorAngles()
#     self.current_motor_angles = self.last_motor_angles

    
#     self.last_base_orientation = env.robot.GetBaseOrientation()
#     self.current_base_orientation = self.last_base_orientation

#   def update(self, env):
#     """Updates the internal state of the task."""
#     self.last_base_pos = self.current_base_pos
#     self.current_base_pos = env.robot.GetBasePosition()
    
#     self.last_motor_angles = self.current_motor_angles
#     self.current_motor_angles = env.robot.GetMotorAngles()

#     self.last_base_orientation = self.current_base_orientation
#     self.current_base_orientation = env.robot.GetBaseOrientation()

#   def done(self, env):
#     """Checks if the episode is over.

#        If the robot base becomes unstable (based on orientation), the episode
#        terminates early.
#     """
#     rot_quat = env.robot.GetBaseOrientation()
#     rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
#     # print(rot_mat)
#     return rot_mat[-1] < 0.85

#     # my more lenient done function
#     # rot_quat = env.robot.GetBaseOrientation()
#     # euler_angles = env.pybullet_client.getEulerFromQuaternion(rot_quat)
#     # print(euler_angles)
#     # diff_angles = np.array(euler_angles) - np.array([math.pi / 2.0, 0, math.pi / 2.0])
#     # return abs(diff_angles[0]) > (math.pi / 2)



#   def reward(self, env):
#     """Get the reward without side effects."""

#     uid = env.robot.quadruped
#     pyb = env._pybullet_client

#     # root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(uid)
#     # reward = np.array(root_vel_sim).dot(np.array([1,0,0]))

#     base_velocity,_ = pyb.getBaseVelocity(uid)
#     z = self.current_base_pos[2]
#     diff_motor_angles = self.current_motor_angles-self.last_motor_angles
#     diff_position = np.array(self.current_base_pos)-np.array(self.last_base_pos)
#     diff_orientation = np.array(self.current_base_orientation)-np.array(self.last_base_orientation)
    
#     # Stay still
#     # seems like a valuable curriculum
#     # reward = np.exp(-10*(np.dot(self.current_base_pos,self.current_base_pos)))

#     # minimize energy v1
#     # minimize energy through minimizing differences of motor angles
#     # energy_reward = np.exp(-25*np.dot(diff_motor_angles,diff_motor_angles))
#     # motor angle differences range 0..0.2 and there are 12 of them.    
    

#     # Minimize Energy
#     E = 10 * np.exp(-env.robot.GetEnergyConsumptionPerControlStep())

#     # Height
#     H = np.exp(-10000*(z-0.45)**4)-0.05
#     # inspection of mocap indicates z between 0.4 and 0.5 is good.
#     # z~=0.2 is lying down which we don't want.
#     # this provides a reasonably square reward for z above 0.3 and below 0.6
#     # outside of this it is slightly negative

#     # Velocity: 
#     # aim to go 1 m/s
#     # big normal distribution which goes slightly negative outside of 0..1.8 m/s
#     V = np.exp(-5*(base_velocity[0]-1)**2)-0.02

#     # Maintain Orientation
#     # rot_quat = env.robot.GetBaseOrientation()
#     # position, orientation = pyb.getBasePositionAndOrientation(uid)
#     # orientation = pyb.getEulerFromQuaternion(orientation)
#     # diff_orientation = np.array(orientation) - np.array([math.pi / 2.0, 0, math.pi / 2.0])
#     # O = np.exp(-np.dot(diff_orientation, diff_orientation))
#     # initial_orientation = np.array([math.pi / 2.0, 0, math.pi / 2.0])

#     # transform z euler angle rotation so it starts at zero when robot is facing the +x direction
#     position, orientation = pyb.getBasePositionAndOrientation(uid)
#     orientation = pyb.getEulerFromQuaternion(orientation)
#     rot = orientation[2]
#     rot -= math.pi/2 
#     if rot < -math.pi:
#       rot += 2*math.pi 
#     # print(rot)

#     # Orientation reward
#     O = 10 * np.exp(-5*rot**2)-0.05
  
#     print({ "energy": E, "height": H, "orientation": O, "velocity": V })
    
#     reward = E+H+O
#     del env
#     return reward

#     # h = max(0, np.exp(-100*(z-0.42)**2) ) 
#     # H = 2 * ( 1/(10*abs(z-0.42)+1) -1/2) 
#     # spikey height reward at 0.42 which is negative outside of 0.3..0.5

# class SimpleForwardTask(object):
#   """Default empy task."""
#   def __init__(self):
#     """Initializes the task."""
#     self.current_base_pos = np.zeros(3)
#     self.last_base_pos = np.zeros(3)

#   def __call__(self, env):
#     return self.reward(env)

#   def reset(self, env):
#     """Resets the internal state of the task."""
#     self._env = env
#     self.last_base_pos = env.robot.GetBasePosition()
#     self.current_base_pos = self.last_base_pos

#   def update(self, env):
#     """Updates the internal state of the task."""
#     self.last_base_pos = self.current_base_pos
#     self.current_base_pos = env.robot.GetBasePosition()

#   def done(self, env):
#     """Checks if the episode is over.

#        If the robot base becomes unstable (based on orientation), the episode
#        terminates early.
#     """
#     rot_quat = env.robot.GetBaseOrientation()
#     rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
#     return rot_mat[-1] < 0.85


#   def reward(self, env):
#     """Get the reward without side effects."""
#     del env
#     reward = self.current_base_pos[0] - self.last_base_pos[0]
#     return reward

