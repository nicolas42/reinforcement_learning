# Install

It uses python 3.7 and virtualenv. Tested on ubuntu 20.04

    # get python3.7, python3.7 venv and some other packages using
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update

    sudo apt install python3.7
    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
    sudo apt install python3.7-venv

    # make a virtual environment called env37
    python3.7 -m venv env37
    source env37/bin/activate
    # deactivate (when done using the environment)

    # install all the things!
    pip install -r requirements.txt
    git clone https://github.com/openai/baselines.git
    pip install -e baselines





# Do stuff

First activate the virtual environment

    source env37/bin/activate




# Motion Imitation

See the motion_imitation README.md

Train

    python3 motion_imitation/run.py --mode train --motion_file motion_imitation/data/motions/dog_pace.txt --int_save_freq 10000000 --visualize

Test default dog_pace.zip network

    python3 motion_imitation/run.py --mode test



# Hexapod PPO

Bespoke PPO system used previously for curriculum learning.

The system trains by default.  Results are written to a folder in the home directory ~/results/../... .
To test the policy use the flag --test_pol.

It can be run in multiple processes using mpirun.  Running in the background can be done with nohup ... & .
Tensorboard results can be viewed by pointing tensorboard to the output directory of the network (the one mentioned above)
but the files can't be accessed at the same time as they are being written so you'll need to make a copy if you want
to view the model at the same time that it's training.

Train

    python3 run1_hexapod.py

Test

This will test the policy located at ./weights/hex .

    python3 run1_hexapod.py --test_pol


Other stuff

    tensorboard --logdir <dir>

Train using more processes in the background

    nohup mpirun -np 4 --oversubscribe python3 run1_hexapod.py &
    


# Train quadruped using Hexapod PPO


Train

    python3 run6_quadruped.py

Test

    python3 run6_quadruped.py --test_pol


Other stuff

    tensorboard --logdir <dir>

Train using more processes in the background

    nohup mpirun -np 4 --oversubscribe python3 run6_quadruped.py &



# Train quadruped using generic PPO

python run7_my_ppo.py






 