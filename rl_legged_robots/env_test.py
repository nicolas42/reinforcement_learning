import pybullet as p
import pybullet_data
import numpy as np
from bullet_env import HexapodBulletEnv
from robots import Hexapod, Minitaur

# client = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -10)
# p.loadURDF('plane.urdf')

env=HexapodBulletEnv(render=True)

max_target = 0.2
target = 0
n_steps = 0
while True:
    if target < max_target:
        target = n_steps * max_target * 1e-4
        env.step(-target * np.ones(30))
    else:
        break
    n_steps += 1

