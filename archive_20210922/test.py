import os
import pybullet as p
import pybullet_data

p.connect(p.GUI)

# pybullet data path
# /home/sch600/miniconda3/lib/python3.9/site-packages/pybullet_data
# accessible with os.path.join(pybullet_data.getDataPath()


laikagoUid = p.loadURDF("laikago/laikago.urdf",useFixedBase=True)
while True:
    p.stepSimulation()

