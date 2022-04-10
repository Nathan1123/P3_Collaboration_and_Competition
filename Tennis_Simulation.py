from Multi_Agent_Learning import Multi_Agent_Learner

# --------------------------------------------------------
#
# Nathan Goedeke
# Deep Reinforcement Learning - Project 3 (Collaboration and Competition)
# April 9, 2022
#
# Description: This script will initialize an environment 
#              and two agents that will be trained to simulate 
#              playing tennis for as long as possible. It is
#              trained over 1400 episodes to achieve an  
#              average score of 0.5 per 100 episodes.
#
# Usage: Set the hyperparameters below to fine tune the 
#        agent training, then execute the file in a python 
#        interpreter. Alternatively, the Jupyter Notebook 
#        Tennis.ipynb performs the same actions as this 
#        file. For more information, see the README file.
#
# ---------------------------------------------------------

# --- SET PARAMETERS HERE ---
env_name='Tennis_Windows_x86_64/Tennis.exe' # Path to executable
seed=0        # Random seed for learning model. Zero for true randomness

actor_lr = 1e-4         # Learning rate (actor)
critic_lr = 1e-3        # Learning rate (critic)
weight_decay = 0        # Weight decay for critic model
gamma = 0.99            # Discount factor
update_every = 1        # How often to update the network
times_update = 1        # How many times to learn each update
buffer_size = int(1e6)  # Replay buffer size
batch_size = 256        # Minibatch size

n_episodes=1500  # Total number of episodes
max_t=1000       # Number of actions per episode
score_window=100 # Number of episodes to average results
score_thresh=0.5 # Minimum value to end learning at
model_name=['actor_0_weights','critic_0_weights','actor_1_weights','critic_1_weights'] # Name to save fully trained model

x_axis='Episode #' # Labels of results graph
y_axis='Score'
# ---------------------------

# Set up Unity environment
agent_params = [actor_lr, critic_lr, weight_decay, gamma, update_every, times_update, buffer_size, batch_size]
DDPG = Multi_Agent_Learner(env_name, seed, agent_params)
# Train for n episodes
DDPG.maddpg_learn(n_episodes, max_t, model_name, score_window, score_thresh)
input("Training complete. Press any key to display results")
# Display training results
DDPG.display(x_axis, y_axis)
# Close environment
DDPG.terminate()