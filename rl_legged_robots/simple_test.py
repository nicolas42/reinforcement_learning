import pybullet as p
import pybullet_data
import numpy as np
from robots import Hexapod, Minitaur

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF('plane.urdf')

weaver = Hexapod(client, pos=[0., 0., 0.2])
minitaur = Minitaur(pos=[0.5, 0.5, 0.2])

for i in range(100):
    _ = p.stepSimulation()

n_steps = 0
max_target = 0.2
target=0.
while True:
    _ = p.stepSimulation()
    if target < max_target:
        target = n_steps * max_target * 1e-4
        weaver.act(-target * np.ones(30))
    n_steps += 1
