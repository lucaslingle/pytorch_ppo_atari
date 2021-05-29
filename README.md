# pytorch_ppo_atari

Implementation of Proximal Policy Optimization in PyTorch, supporting parallel experience collection. 

![beamrider gif](assets/beamrider-ppo-paper-defaults/beamrider.gif)
![breakout gif](assets/breakout-ppo-paper-defaults/breakout.gif)
![mspacman gif](assets/mspacman-ppo-paper-defaults/mspacman.gif)
![pong gif](assets/model-ppo1-defaults/pong.gif)
![enduro gif](assets/enduro-ppo-paper-defaults/enduro.gif)

## Background

Proximal Policy Optimization is a reinforcement learning algorithm proposed 
by [Schulman et al., 2017](https://arxiv.org/abs/1707.06347). Compared to vanilla policy gradients 
and/or actor-critic methods, which optimize the model parameters by estimating the gradient of the reward surface
and taking a single step, PPO takes inspiration from an approximate natural policy gradient algorithm known as TRPO.

[TRPO](https://arxiv.org/abs/1502.05477) is an example of an information-geometric trust region method, 
which aims to improve the policy by taking steps of a constant maximum size on the manifold of possible policies.
The step-size utilized in TRPO is the state-averaged KL divergence under the current policy; taking steps 
under TRPO amounts to solving a constrained optimization problem to ensure the step size is at most a certain amount. 
This is done using conjugate gradient descent to compute the (approximate) natural gradient, followed by a line search 
to ensure the step taken in parameter space leads to a policy whose state-averaged KL divergence to the previous policy 
is not larger than a certain amount. 

Compared to vanilla policy gradients and/or actor-critic methods, the PPO algorithm enjoys the following favorable 
properties:
- Improved sample efficiency
- Improved stability
- Improved reusability of collected experience

Compared to TRPO, proximal policy optimization is considerably simpler, easier to implement, and allows recurrent 
policies without any additional complication. Proximal policy optimization has been used in a number of high-profile 
projects, such as [OpenAI Five](https://arxiv.org/abs/1912.06680) and [Solving a Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113). 

## Getting Started

Install the following system dependencies:
#### Ubuntu     
```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of the system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Everyone
Once the system dependencies have been installed, it's time to install the python dependencies. 
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

Then run
```bash
conda create --name ppo python=3.8.1
conda activate ppo
git clone https://github.com/lucaslingle/ppo
cd ppo
pip install -e .
```

## Usage

### Training
To run the default settings, you can simply type:
```bash
mpirun -np 8 python -m main --env_name=PongNoFrameskip-v4
```

This will launch 8 parallel processes, each running the ```main.py``` script. These processes will play the OpenAI gym environment 'PongNoFrameskip-v4' in parallel, 
and communicate gradient information and synchronize parameters using [OpenMPI](https://www.open-mpi.org/).

To see additional options, you can simply type ```python main.py --help```. In particular, you can pick any other Atari 2600 game supported by OpenAI gym, 
and this implementation will support it. 

### Checkpoints
By default, checkpoints are saved to ```./checkpoints/model-ppo1-defaults```. To pick a different checkpoint directory, 
you can set the ```--checkpoint_dir``` flag, and to pick a different checkpoint name, you can set the ```--model_name``` flag.

### Play
To watch the trained agent play a game, you can run
```bash
mpirun -np 1 python -m main --env_name=PongNoFrameskip-v4 --mode=play
```
Be sure to use the correct env_name, and to pass in the appropriate checkpoint_dir and model_name.

### Video
We also support video recordings of the trained agent. These can be created by running
```bash
mpirun -np 1 python -m main --env_name=PongNoFrameskip-v4 --mode=record
```
Be sure to use the correct env_name, and to pass in the appropriate checkpoint_dir and model_name.
By default, videos are saved to the directory 'assets'. To specify a custom directory, you can set the 
```--asset_dir``` flag.

### Notes
This project started out as a Pytorch port of OpenAI baselines ppo1, and the legacy repo for that port is 
available [here](https://github.com/lucaslingle/ppo1). Since then, we have rewritten the entire implementation 
from scratch, and simplified it by removing all dependencies on OpenAI baselines' code.

In the future, we would like to implement training pipelines for other RL algorithms such as DQN and PER, 
and to add support for recurrent policies like LSTMs, SNAILs and Transformers. These recurrent architectures may also  
benefit from special environments in order to provide a value-add, so we have deferred such an implementation 
to future projects.
