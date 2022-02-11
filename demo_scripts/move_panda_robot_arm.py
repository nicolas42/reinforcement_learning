import os
import pybullet as p
import pybullet_data
import math
import numpy as np 

p.connect(p.GUI)
urdfRootPath=pybullet_data.getDataPath()
pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)

tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

p.setGravity(0,0,-10)
objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=[0.7,0,0.1])

p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

state_durations = [0.5,0.5,0.5,0.5]
control_dt = 1./240.
p.setTimestep = control_dt
state_t = 0.
current_state = 0

numJoints = p.getNumJoints(pandaUid)
print(numJoints)
for i in range(numJoints):
    print(p.getJointInfo(pandaUid,i))

# (0, b'panda_joint1', 0, 7, 6, 1, 0.0, 0.0, -2.9671, 2.9671, 87.0, 2.175, b'panda_link1', (0.0, 0.0, 1.0), (0.0, 0.0, 0.28300000000000003), (0.0, 0.0, 0.0, 1.0), -1)
# (1, b'panda_joint2', 0, 8, 7, 1, 0.0, 0.0, -1.8326, 1.8326, 87.0, 2.175, b'panda_link2', (0.0, 0.0, 1.0), (0.0, 0.04, 0.05), (0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 0)
# (2, b'panda_joint3', 0, 9, 8, 1, 0.0, 0.0, -2.9671, 2.9671, 87.0, 2.175, b'panda_link3', (0.0, 0.0, 1.0), (0.0, -0.276, -0.06), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 1)
# (3, b'panda_joint4', 0, 10, 9, 1, 0.0, 0.0, -3.1416, 0.0, 87.0, 2.175, b'panda_link4', (0.0, 0.0, 1.0), (0.07250000000000001, -0.01, 0.05), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 2)
# (4, b'panda_joint5', 0, 11, 10, 1, 0.0, 0.0, -2.9671, 2.9671, 12.0, 2.61, b'panda_link5', (0.0, 0.0, 1.0), (-0.052500000000000005, 0.354, -0.02), (0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 3)
# (5, b'panda_joint6', 0, 12, 11, 1, 0.0, 0.0, -0.0873, 3.8223, 12.0, 2.61, b'panda_link6', (0.0, 0.0, 1.0), (0.0, -0.04, 0.12), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 4)
# (6, b'panda_joint7', 0, 13, 12, 1, 0.0, 0.0, -2.9671, 2.9671, 12.0, 2.61, b'panda_link7', (0.0, 0.0, 1.0), (0.047999999999999994, 0.0, 0.0), (-0.7071067811848163, 0.0, 0.0, 0.7071067811882787), 5)
# (7, b'panda_joint8', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'panda_link8', (0.0, 0.0, 0.0), (0.0, 0.0, 0.026999999999999996), (0.0, 0.0, 0.0, 1.0), 6)
# (8, b'panda_hand_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'panda_hand', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.38268343236488267, 0.9238795325113726), 7)
# (9, b'panda_finger_joint1', 1, 14, 13, 1, 0.0, 0.0, 0.0, 0.04, 20.0, 0.2, b'panda_leftfinger', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0184), (0.0, 0.0, 0.0, 1.0), 8)
# (10, b'panda_finger_joint2', 1, 15, 14, 1, 0.0, 0.0, 0.0, 0.04, 20.0, 0.2, b'panda_rightfinger', (0.0, -1.0, 0.0), (0.0, 0.0, 0.0184), (0.0, 0.0, 0.0, 1.0), 8)
# (11, b'panda_grasptarget_hand', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'panda_grasptarget', (0.0, 0.0, 0.0), (0.0, 0.0, 0.065), (0.0, 0.0, 0.0, 1.0), 8)


# The pose (position and orientation) of the end-effector of the robot can be read using pybullet.getLinkState().
# Joint variables of the fingers can be read using pybullet.getJointState()

# There's 12 joints
# last joint is end effector

# joint position is an angle


while True:
    state_t += control_dt
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    if current_state == 0:
        p.setJointMotorControlArray(pandaUid, [0,1,2,3,4,5,6,9,10], p.POSITION_CONTROL, [0, math.pi/4., 0, -math.pi/2., 0, 3*math.pi/4, -math.pi/4., 0.08, 0.08])
        # p.setJointMotorControl2(pandaUid, 0, p.POSITION_CONTROL,0)
        # p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL,math.pi/4.)
        # p.setJointMotorControl2(pandaUid, 2, p.POSITION_CONTROL,0)
        # p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL,-math.pi/2.)
        # p.setJointMotorControl2(pandaUid, 4, p.POSITION_CONTROL,0)
        # p.setJointMotorControl2(pandaUid, 5, p.POSITION_CONTROL,3*math.pi/4)
        # p.setJointMotorControl2(pandaUid, 6, p.POSITION_CONTROL,-math.pi/4.)
        # p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, 0.08)
        # p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, 0.08)
    if current_state == 1:
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL,math.pi/4.+.15)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL,-math.pi/2.+.15)
    if current_state == 2:
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, 0.0, force = 200)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, 0.0, force = 200)
    if current_state == 3:
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL,math.pi/4.-1)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL,-math.pi/2.-1)

    if state_t >state_durations[current_state]:
        current_state += 1
        if current_state >= len(state_durations):
            current_state = 0
        state_t = 0
        
    link_state = p.getLinkState( pandaUid, 11, computeForwardKinematics = True )
    panda_position, panda_orientation = link_state[0], link_state[1]
    object_position, object_orientation = p.getBasePositionAndOrientation(objectUid)
    distance_between_panda_fingers_and_object = np.sqrt(np.sum(np.square(np.array(panda_position)-np.array(object_position))))
    # print(distance_between_panda_fingers_and_object)
    observation = p.getLinkStates(pandaUid, [0,1,2,3,4,5,6,7,8,9,10,11], computeForwardKinematics = True )
    # print(observation[11][0])

    joint_states = p.getJointStates(pandaUid, [0,1,2,3,4,5,6,7,8,9,10,11] )
    joint_states = [ joint_state[0] for joint_state in joint_states ]
    print(joint_states)

    p.stepSimulation()





