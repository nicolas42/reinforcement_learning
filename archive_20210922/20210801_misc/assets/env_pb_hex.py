import pybullet as p
import numpy as np
import os   
import time
from mpi4py import MPI
from collections import deque
from pathlib import Path
home = str(Path.home())
comm = MPI.COMM_WORLD
  
class Env():
    rank = comm.Get_rank()
    ob_size = 56
    ac_size = 18
    im_size = [60,40,1]
    timeStep = 1/120
    total_steps = 0
    episodes = -1
    def __init__(self, PATH=None, args=None):
        self.render = args.render
        self.PATH = PATH
        self.horizon = 2000
        self.record_step = args.record_step
        self.cur = args.cur
        self.args = args
        if self.render:
            self.physicsClientId = p.connect(p.GUI)
        else:
            self.physicsClientId = p.connect(p.DIRECT) #DIRECT is much faster, but GUI shows the robot
        self.load_model()
        self.sim_data = []

    def load_model(self):
        hm = np.loadtxt("assets/Terrain_1.txt", delimiter=",")
        # squash heightmap to 1D array
        heightmap_flat = np.flip(hm, 0).reshape(-1) * 0

        meshscale = [0.05, 0.05, 1]
        rows, cols = hm.shape

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                              heightfieldData=heightmap_flat,
                                              meshScale=meshscale,
                                              numHeightfieldRows=rows, numHeightfieldColumns=cols)

        ter = p.createMultiBody(0, terrainShape)

        p.resetBasePositionAndOrientation(ter, [0, 0, -0.2], p.getQuaternionFromEuler([0, 0, 0]))
        #p.changeDynamics(ter, -1, lateralFriction=0.9, spinningFriction=0.9, rollingFriction=0.9)
        # p.loadMJCF("./assets/ground.xml")
        objs = p.loadURDF("./assets/hexapod.urdf")
        self.Id = objs
        p.setTimeStep(self.timeStep)
        p.setGravity(0,0,-9.8)
        self.feet_dict = {}
        self.feet = ["AR_foot_link","AL_foot_link","BR_foot_link",
                    "BL_foot_link","CR_foot_link","CL_foot_link"]
        numJoints = p.getNumJoints(self.Id)
        self.jdict = {}
        self.ordered_joints = []
        self.ordered_joint_indices = []

        for j in range( p.getNumJoints(self.Id) ):
            info = p.getJointInfo(self.Id, j)
            link_name = info[12].decode("ascii")
            if link_name in self.feet: self.feet_dict[link_name] = j
            self.ordered_joint_indices.append(j)
            if info[2] != p.JOINT_REVOLUTE: continue
            jname = info[1].decode("ascii")
            lower, upper = (info[8], info[9])
            self.ordered_joints.append( (j, lower, upper) )
            # print(jname)
            self.jdict[jname] = j

        self.state = [  "AR_coxa_joint_pos","AR_femur_joint_pos","AR_tibia_joint_pos",
                        "AL_coxa_joint_pos","AL_femur_joint_pos","AL_tibia_joint_pos",
                        "BR_coxa_joint_pos","BR_femur_joint_pos","BR_tibia_joint_pos",
                        "BL_coxa_joint_pos","BL_femur_joint_pos","BL_tibia_joint_pos",
                        "CR_coxa_joint_pos","CR_femur_joint_pos","CR_tibia_joint_pos",
                        "CL_coxa_joint_pos","CL_femur_joint_pos","CL_tibia_joint_pos"]
        self.state += [ "AR_coxa_joint_vel","AR_femur_joint_vel","AR_tibia_joint_vel",
                        "AL_coxa_joint_vel","AL_femur_joint_vel","AL_tibia_joint_vel",
                        "BR_coxa_joint_vel","BR_femur_joint_vel","BR_tibia_joint_vel",
                        "BL_coxa_joint_vel","BL_femur_joint_vel","BL_tibia_joint_vel",
                        "CR_coxa_joint_vel","CR_femur_joint_vel","CR_tibia_joint_vel",
                        "CL_coxa_joint_vel","CL_femur_joint_vel","CL_tibia_joint_vel"]

        self.state += ['vx','vz','vy','roll','pitch','pitch_vel','roll_vel','yaw_vel']

        # Tip contacts
        self.state += ["AR_foot_link_left_ground","AL_foot_link_left_ground",       "BR_foot_link_left_ground",
                        "BL_foot_link_left_ground","CR_foot_link_left_ground","CL_foot_link_left_ground"]
        self.state += ["prev_AR_foot_link_left_ground","prev_AL_foot_link_left_ground","prev_BR_foot_link_left_ground",
                        "BL_foot_link_left_ground","prev_CR_foot_link_left_ground","prev_CL_foot_link_left_ground"]
        
        # self.state += ['left_foot_on_ground', 'right_foot_on_ground']
        # self.state += ['swing_foot']
        # self.state += ['com_z']


        self.motor_names = [
                            "AR_coxa_joint","AR_femur_joint","AR_tibia_joint",
                            "AL_coxa_joint","AL_femur_joint","AL_tibia_joint",
                            "BR_coxa_joint","BR_femur_joint","BR_tibia_joint",
                            "BL_coxa_joint","BL_femur_joint","BL_tibia_joint",
                            "CR_coxa_joint","CR_femur_joint","CR_tibia_joint",
                            "CL_coxa_joint","CL_femur_joint","CL_tibia_joint"]


        self.motors = [self.jdict[n] for n in self.motor_names]

        self.motor_power =  [15, 22, 15]
        self.motor_power += [15, 22, 15]
        self.motor_power += [15, 22, 15]
        self.motor_power += [15, 22, 15]
        self.motor_power += [15, 22, 15]
        self.motor_power += [15, 22, 15]
        self.motor_power = np.array(self.motor_power)
        self.tor_max =  [80,112,80]
        self.tor_max += [80,112,80]
        self.tor_max += [80,112,80]
        self.tor_max += [80,112,80]
        self.tor_max += [80,112,80]
        self.tor_max += [80,112,80]
        self.tor_max = np.array(self.tor_max)

        forces = np.ones(len(self.motors))*240
        self.actions = {key:0.0 for key in self.motor_names}
        self.initial_joints = [0.4,  0.0, 1.7]
        self.initial_joints += [-0.4,0.0, 1.7]
        self.initial_joints += [0.0, 0.0, 1.7]
        self.initial_joints += [0.0, 0.0, 1.7]
        self.initial_joints += [-0.4, 0.0, 1.7]
        self.initial_joints += [0.4,0.0, 1.7]
        self.target_height = 0.3
        self.initial_height = 0.4

        # Disable motors to use torque control:
        p.setJointMotorControlArray(self.Id, self.motors, controlMode=p.VELOCITY_CONTROL, forces=[0.] * len(self.motor_names))

        # Increase the friction on the feet.
        for key in self.feet_dict:
            p.changeDynamics(self.Id, self.feet_dict[key],lateralFriction=0.9, spinningFriction=0.9, rollingFriction=0.1)

    def close(self):
        print("closing")
  
    def reset(self, cur_params=None):

        self.total_reward = 0
        self.steps = 0
        self.episodes += 1

        self.set_position([0,0,self.initial_height], [0,0,0,1], joints=self.initial_joints, velocities=[[0,0,0],[0,0,0]], joint_vel=[0.0]*self.ac_size)

        self.ob_dict = {}
        for n in self.feet_dict:
            self.ob_dict['prev_' + n + '_left_ground'] = 0
            self.ob_dict[n + '_left_ground']= 0

        self.get_observation()
        if self.record_step:
            self.save_sim_data()
            self.sim_data = []
            
        return np.array([self.ob_dict[s] for s in self.state])

    def step(self, actions, set_position=None):
        if set_position is not None:
            self.set_position(set_position[0], set_position[1], joints=set_position[2])
        else:
            torques = np.clip(actions*self.motor_power, -self.tor_max, self.tor_max)
            p.setJointMotorControlArray(self.Id, self.motors,controlMode=p.TORQUE_CONTROL, forces=torques)  
        p.stepSimulation()
        # if self.render:
        # time.sleep(0.01)
        self.get_observation()
        reward, done = self.get_reward(actions)
        self.total_reward += reward
        if self.record_step: 
            self.record_sim_data()
        if self.args.render:
            time.sleep(0.01)
        self.steps += 1
        self.total_steps += 1
        return np.array([self.ob_dict[s] for s in self.state]), reward, done, self.ob_dict

    def get_reward(self, actions=None):
        reward = 0
        # Don't penalise going faster than 1.0
        reward = np.exp(-2.5*max(0, 1.0 - self.ob_dict["vx"])**2)
        # Penalty for body orientation and height
        # body_error = [(self.ob_dict["roll"] - 0)**2, (self.ob_dict["roll"] - 0)**2, (self.ob_dict["yaw"] - 0)**2, (self.ob_dict["y"] - 0)**2, (self.ob_dict["z"] - self.target_height)**2]
        body_error = [(self.ob_dict["roll"] - 0)**2, (self.ob_dict["roll"] - 0)**2, (self.ob_dict["yaw"] - 0)**2, (self.ob_dict["z"] - self.target_height)**2]
        reward -= np.sum(body_error)
        # Try and keep the joints close to the initial joint positions (stop joints positions being wild)
        # reward -= 0.001*np.sum((np.array(self.joints) - np.array(self.initial_joints))**2)
        # Penalty for large actions
        # reward -= 0.0001*np.sum(np.array(actions)**2)
        # Penalty for having more than 3 feet off the ground at once (otherwise the robot jumps, try to somewhat encourage walking)
        # num_off_the_ground = sum([self.ob_dict[n + '_left_ground'] for n in self.feet_dict])
        # if num_off_the_ground > 2:
            # reward -= 0.01*num_off_the_ground
        done = False
        # if self.steps > self.horizon or self.ob_dict["z"] < 0.25 or (np.abs([self.ob_dict["roll"], self.ob_dict["pitch"]]) > 0.25).any():
        # Stop the run if the z height is below a threshold, or the pitch or roll is too high
        if not self.args.test_pol and (self.steps > self.horizon or self.ob_dict["z"] < 0.25 or (np.abs([self.ob_dict["roll"], self.ob_dict["pitch"], self.ob_dict["yaw"]]) > 0.35).any()):
            done = True
        return reward, done
 
    def get_observation(self):

        jointStates = p.getJointStates(self.Id,self.ordered_joint_indices)
        self.joints = [jointStates[j[0]][0] for j in self.ordered_joints[:int(self.ac_size)]]
        self.joint_vel = [jointStates[j[0]][1] for j in self.ordered_joints[:int(self.ac_size)]]

        for n in self.feet_dict:
            self.ob_dict['prev_' + n + '_left_ground'] = self.ob_dict[n + '_left_ground']
            self.ob_dict[n + '_left_ground']= not len(p.getContactPoints(self.Id, -1, self.feet_dict[n], -1))>0
        
        self.ob_dict.update({self.state[i]:jointStates[j[0]][0] for i,j in enumerate(self.ordered_joints[:int(18)])})
        self.ob_dict.update({self.state[i+18]:jointStates[j[0]][1] for i,j in enumerate(self.ordered_joints[:int(18)])})
        
        [x,y,z], (qx, qy, qz, qw) = p.getBasePositionAndOrientation(self.Id)
        roll, pitch, yaw = p.getEulerFromQuaternion([qx, qy, qz, qw])
        body_vxyz, [roll_vel, pitch_vel, yaw_vel] = p.getBaseVelocity(self.Id)

        rot_speed = np.array(
        [[np.cos(-yaw), -np.sin(-yaw), 0],
            [np.sin(-yaw), np.cos(-yaw), 0],
            [		0,			 0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, (body_vxyz[0],body_vxyz[1],body_vxyz[2]))  

        self.ob_dict.update({"x":x, "y":y, "z":z})
        self.ob_dict.update({"qx":qx, "qy":qy, "qz":qz, "qw":qw})
        self.ob_dict.update({'vx':vx,'vy':vy,'vz':vz})
        self.ob_dict.update({'roll':roll,'pitch':pitch,'yaw':yaw})
        self.ob_dict.update({"roll_vel":roll_vel, "pitch_vel":pitch_vel, "yaw_vel":yaw_vel})

    def set_position(self, pos, orn, joints=None, velocities=None, joint_vel=None, robot_id=None):
        if robot_id is None:
            robot_id = self.Id
        pos = [pos[0], pos[1], pos[2]]
        p.resetBasePositionAndOrientation(robot_id, pos, orn)
        if joints is not None:
            if joint_vel is not None:
                for j, jv, m in zip(joints, joint_vel, self.motors):
                    p.resetJointState(robot_id, m, targetValue=j, targetVelocity=jv)
            else:
                for j, m in zip(joints, self.motors):
                    p.resetJointState(robot_id, m, targetValue=j)
        if velocities is not None:
            p.resetBaseVelocity(robot_id, velocities[0], velocities[1])

    def get_im(self):
        return np.zeros(self.im_size)

    def save_sim_data(self):
        if self.rank == 0:
            try:
                    np.save(self.PATH + 'sim_data.npy', np.array(self.sim_data))
            except Exception as e:
                print("Save sim data error:")
                print(e)

    def record_sim_data(self):
        if len(self.sim_data) > 100000: return
        pos, orn = p.getBasePositionAndOrientation(self.Id)
        data = [pos, orn]
        joints = p.getJointStates(self.Id, self.motors)
        data.append([i[0] for i in joints])
        self.sim_data.append(data)

    def log_stuff(self, logger, writer, iters_so_far):
        pass