import sys
import pybullet as p
import time
import numpy as np
from numpy import cos as c
from numpy import sin as s
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render', default=True, action='store_false')
parser.add_argument('--euler', default=True, action='store_false')
parser.add_argument('--fixed', default=True, action='store_false')
parser.add_argument('--torque', default=True, action='store_false')
parser.add_argument('--record_step', default=False, action='store_true')
parser.add_argument('--cur', default=False, action='store_true')
parser.add_argument('--robot', default='biped')
args = parser.parse_args()

if args.robot == "hexapod":
	from assets.env_pb_hex import Env
else:
	from assets.env_pb_biped import Env

env = Env(PATH=None, args=args)

obs = env.reset()
if args.euler:
	angles = ['roll', 'pitch', 'yaw']
	angle_target = {name: {'target': p.addUserDebugParameter(name + "_target",-2.0,2.0,0)} for name in angles}
else:
	angles = ['qx', 'qy', 'qz','qw']
	angle_target = {name: {'target': p.addUserDebugParameter(name + "_target",-1.0,1.0,0)} for name in angles}
if args.torque:
	joint_target = {name: {'target': p.addUserDebugParameter(name + "_target",-75.0,75.0,0)} for name in env.motor_names}
else:
	joint_target = {name: {'target': p.addUserDebugParameter(name + "_target",j[1],j[2],0)} for name, j in zip(env.motor_names, env.ordered_joints)}

pos = ['x', 'y', 'z']
pos_target = {name: {'target': p.addUserDebugParameter(name + "_target",-1.5,1.5,0)} for name in pos if name in ['x', 'y']}
pos_target['z'] = {'target': p.addUserDebugParameter("z_target",-0.5,2.5,1.0)}

# cam_target = {}
# cam_target['dx'] = {'target': p.addUserDebugParameter("dx_target",-1.5,1.5,0.6)}
# cam_target['dy'] = {'target': p.addUserDebugParameter("dy_target",-1.5,1.5,0.0)}
# cam_target['dz'] = {'target': p.addUserDebugParameter("dz_target",-1.5,1.5,0.-0.05)}
# cam_target['cam_x'] = {'target': p.addUserDebugParameter("cam_x_target",-1.5,1.5,0.45)}
# cam_target['cam_y'] = {'target': p.addUserDebugParameter("cam_y_target",-1.5,1.5,0.0)}
# cam_target['cam_z'] = {'target': p.addUserDebugParameter("cam_z_target",-1.5,1.5,0.0)

actions = np.zeros(env.ac_size)
while (1):
	time.sleep(0.01)
	if args.euler:
		roll = p.readUserDebugParameter(angle_target['roll']['target'])
		pitch = p.readUserDebugParameter(angle_target['pitch']['target'])
		yaw = p.readUserDebugParameter(angle_target['yaw']['target'])
	else:
		qx = p.readUserDebugParameter(angle_target['qx']['target'])
		qy = p.readUserDebugParameter(angle_target['qy']['target'])
		qz = p.readUserDebugParameter(angle_target['qz']['target'])
		qw = p.readUserDebugParameter(angle_target['qw']['target'])

	x = p.readUserDebugParameter(pos_target['x']['target'])
	y = p.readUserDebugParameter(pos_target['y']['target'])
	z = p.readUserDebugParameter(pos_target['z']['target'])
	
	# dx = p.readUserDebugParameter(cam_target['dx']['target'])
	# dy = p.readUserDebugParameter(cam_target['dy']['target'])
	# dz = p.readUserDebugParameter(cam_target['dz']['target'])
	# cam_x = p.readUserDebugParameter(cam_target['cam_x']['target'])
	# cam_y = p.readUserDebugParameter(cam_target['cam_y']['target'])
	# cam_z = p.readUserDebugParameter(cam_target['cam_z']['target'])
	if not args.euler:
		# orn = [0,0,-0.707, 0.707]
		orn = [qx, qy, qz, qw]
	else:
		orn = p.getQuaternionFromEuler([roll, pitch, yaw])

	actions = [0]*env.ac_size
	for i,j in enumerate(env.motor_names):
		actions[i] = p.readUserDebugParameter(joint_target[j]['target'])
		
	# env.set_position([0,0,1.0], orn)
	env.set_position([x,y,z], orn)

	if args.torque:
		p.setJointMotorControlArray(env.Id, env.motors,controlMode=p.TORQUE_CONTROL, forces=actions)
	else:
		p.setJointMotorControlArray(env.Id, env.motors,controlMode=p.POSITION_CONTROL, targetPositions=actions)

	p.stepSimulation()

