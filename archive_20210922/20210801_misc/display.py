'''
Script used to display robot stuff while training (loads in save robot state)
'''
import pybullet as p
import numpy as np
import argparse
from pathlib import Path
import time
import os
home = str(Path.home())

# Hack for loading defaults, but still accepting run specific defaults
import defaults
from dotmap import DotMap
args1, unknown1 = defaults.get_defaults() 
parser = argparse.ArgumentParser()
# Arguments that are specific for this run (including run specific defaults, ignore unknown arguments)
parser.add_argument('--folder', default='hex')
parser.add_argument('--exp', default='test')
parser.add_argument('--robot', default='hexapod', help='biped or hexapod')
parser.add_argument('--render', default=True, action='store_false')
parser.add_argument('--record_step', default=False, action='store_true')
args2, unknown2 = parser.parse_known_args()
args2 = vars(args2)
# Replace any arguments from defaults with run specific defaults
for key in args2:
    args1[key] = args2[key]
# Look for any changes to defaults (unknowns) and replace values in args1
for n, unknown in enumerate(unknown2):
    if "--" in unknown and n < len(unknown2)-1 and "--" not in unknown2[n+1]:
        arg_type = type(args1[unknown[2:]])
        args1[unknown[2:]] = arg_type(unknown2[n+1])
args = DotMap(args1)
# Check for dodgy arguments
unknowns = []
for unknown in unknown1 + unknown2:
    if "--" in unknown and unknown[2:] not in args:
        unknowns.append(unknown)
if len(unknowns) > 0:
    print("Dodgy argument")
    print(unknowns)
    exit()

if args.hpc:
    hpc = 'hpc-home/'
else:
    hpc = ''
MODEL_PATH = home + '/' + hpc + 'results/hexapod/latest/' + args.exp + '/'

from assets.env_pb_hex import Env

env = Env(PATH=None, args=args)

def data_gen(path, box_path = None):
    print(path)
    env.reset()
    while True:
        try:
            try:
                box_info = np.load(box_path, allow_pickle=True)
            except:
                box_info = None
            sim_data = np.load(path, allow_pickle=True)
            if box_info is not None:
                env.reset(box_info=box_info)
            else:
                env.reset()
            if len(sim_data) == 0:
                yield None
            count = 0
            for seg in sim_data:
                yield seg, count
                count += 1
        except Exception as e:
            print("exception")
            print(e)
data = data_gen(path = MODEL_PATH + 'sim_data.npy', box_path = MODEL_PATH + 'box_info.npy')

while True:
    seg, count = data.__next__()
    if seg is not None:
        env.step(actions=np.zeros(env.ac_size), set_position=seg)
