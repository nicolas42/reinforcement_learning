# README #

### What is this repository for? ###

* Curriculum  learning code for guided curriculum learning 
* RL for DSH1. The algorithm used is Proximal Policy Optimisation (PPO), adapted from https://github.com/openai/baselines


### Curriculum Info: ###
- Curriculum is evaluated every episode. If the curriculum is reached (3 successes in a row) the current curriculum is adjusted. This exists in the reset function of the environment: i.e. def reset() in env_pb_biped.py
- Three parts:
1. Guide forces: variable = env.Kp (range = 400 - 0), default == 400 (if python3 run.py --cur)
2. Terrain difficulty: variable = env.difficulty (range = 1-10), default = 1
3. Perturbations: variable = env.max_disturbance (range = 50 - 2000), default == 50


### How do I get set up? ###

#### Dependencies: ####
- pybullet (for pybullet sim) </br>
- tensorflow==1.14  </br>
- baselines (from https://github.com/openai/baselines) </br>
- Probably others.. </br>

#### To train: ####

`mpirun -np 16 --oversubscribe python3 run.py --exp my_test --folder test_folder` </br>
- However many cores you want to use 
- will save tensorboard plots etc to /home/User/results/rl_hex/latest/test_folder/my_test</br>

#### During training: ####
* Display while training (run at anytime, will show training in pybullet):</br>
 `python3 display.py --exp test_folder/my_test`</br>
* Tensorboard:</br>
`tensorboard --logdir /home/User/results/rl_hex/latest/test_folder` </br>

#### To test: ####
`python3 run.py --exp my_test --folder test_folder --test_pol`</br>

#### Useful flags ####
- --render render world





# My Addendum


### How do I get set up? ###

Install instructions using python 3.7 and virtualenv


    # ppa for python3.7
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update


    sudo apt install python3.7
    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
    sudo apt install python3.7-venv

    python3.7 -m venv env37
    source env37/bin/activate
    # deactivate (when done using the environment)

    pip install -r requirements.txt

    git clone https://github.com/openai/baselines.git
    pip install -e baselines



#### To train: ####

    mpirun -np 8 --oversubscribe python3 run.py --exp my_test --folder test_folder

- However many cores you want to use 
- will save tensorboard plots etc to /home/User/results/rl_hex/latest/test_folder/my_test

#### During training: ####
* Display while training (run at anytime, will show training in pybullet):

    python3 display.py --exp test_folder/my_test

* Tensorboard:

    tensorboard --logdir /home/User/results/rl_hex/latest/test_folder 

#### To test: ####

    python3 run.py --exp my_test --folder test_folder --test_pol --render render world

#### Useful flags ####
- --render render world






