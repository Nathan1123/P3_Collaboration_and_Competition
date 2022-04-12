[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"


# Project 3: Collaboration and Competition

### Overview

In this project, an intelligent agent will be trained to use the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net, simulating the game of tennis. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, both agents are trained to keep the ball in play for as long as possible.

The state space has 8 dimensions corresponding to the position and velocity of the ball and each racket. The agent uses this information to select the next best action. Each agent recieves state obsrvations independently, and selects an action independently. Each action is a continuous value closer or farther away from the net, or jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 over 100 consecutive episodes (taking the maximum reward of the two agents per episode). Specifically,

- After each episode, add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. Then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Prerequisites

The following instructions should be completed prior to running the code:

0. Ensure the following are already installed:
* Python 3.7 or higher
  * numpy, matplotlib, unityagents, torch
* Anaconda
  * Jupyter notebooks
  * Git

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. To install the base Gym library, use `pip install gym`. Supports Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. 
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

7. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

8. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### File descriptions

- model.py - defines neural network used to train the agent
- buffer.py - defines replay buffer used during training
- noise.py - random generation of noise durign training
- ddpg_agent.py - defines agent, with functions to act, learn and remember previous states in a replay buffer
- Multi_Agent_Learning.py - sets up an agent in an environment, trains it, and displays results
- Tennis_Simulation.py - sets hyperparameters and uses MADDPG_Learning to train an agent in the Reacher environment 
- Tennis.ipynb - Jupyter notebook to run the training and simulation

### Instructions

The following instructions will run the environment simulation, train the agent and display results:

1. Open Anaconda and navigate to the folder containing Navigation.ipynb. Run the following commands to open the notebook:
    ```bash
    conda activate drlnd
	jupyter notebook Tennis.ipynb
    ```
2. The usable code starts in block 7, and all other blocks can be ignored. The hyperparameters can be altered as the user sees fit.
