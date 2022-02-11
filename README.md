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





# Activate Environment

First activate the virtual environment

    source env37/bin/activate



# Reinforcement learning

See "output" directory for trained model examples

demo

run.py --demo input_file

(runs in deterministic mode)

example:

python run.py --demo output/latest.zip

tip: use alias p="run.py --demo output/latest.zip" and then use p <input file>


train

run.py input_file output_dir

set input_file to None to start from scratch


Most models use [64,64] hidden layer architectures but some use [512,256].  Change the global kwargs value in run.py to use them.




# Motion Imitation

See the README.md in the motion_imitation directory for details

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
    