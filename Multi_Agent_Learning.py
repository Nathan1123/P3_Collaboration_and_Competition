from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch
from ddpg_agent import *

class Multi_Agent_Learner:
    # Initialize environment and agent decision model
    def __init__(self, env_name, agent_seed, agent_params):
        self.env = UnityEnvironment(file_name=env_name)
        # Agent actions
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]
        action_size = brain.vector_action_space_size
        # Agent states
        env_info = self.env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        state_size = states.shape[1]
        # Simulation agents
        self.agents = []
        self.agents.append(Agent(state_size=state_size, action_size=action_size, random_seed=agent_seed, params=agent_params))
        self.agents.append(Agent(state_size=state_size, action_size=action_size, random_seed=agent_seed, params=agent_params))
        self.scores = []
        print("NOTE: The environment will wait until maddpg_learn is run")
        
    # Function to train agent and record scores
    # This can be run any number of times before terminating the environment
    def maddpg_learn(self, n_episodes, max_t, model_name, score_window, score_thresh):
        scores = []
        brain_name = self.env.brain_names[0]
        # Loop per episode
        for i_episode in range(1, n_episodes+1):
            env_info = self.env.reset(train_mode=True)[brain_name] # Reset environment
            states = env_info.vector_observations
            score = np.zeros(2)
            # Actions per episode
            for t in range(max_t):
                action0 = np.array(self.agents[0].act(torch.from_numpy(states[0]).float()))
                action1 = np.array(self.agents[1].act(torch.from_numpy(states[1]).float()))
                env_info = self.env.step([action0, action1])[brain_name]
                next_states = np.array(env_info.vector_observations)
                dones = np.array(env_info.local_done)
                rewards = np.array(env_info.rewards)
                Agent.add_memory(states[0], action0, rewards[0], next_states[0], dones[0],
                             states[1], action1, rewards[1], next_states[1], dones[1])
                self.agents[0].step(0)
                self.agents[1].step(1)
                score += rewards
                states = next_states
            
                # If episode ends, break
                if any(dones):
                    break 
            # Statistics at end of episode
            scores.append(np.max(score))
            avg_score = np.mean(scores[-score_window:]) # Average score
            max_score = np.max(scores[-score_window:]) # Maximum score
            # Print results
            print('\rEpisode {} \tAverage Score: {:.2f} \tMax score: {}'.format(i_episode, avg_score, max_score), end="")
            if i_episode % score_window == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            if avg_score >= score_thresh and i_episode > score_window:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, avg_score))
                torch.save(agents[0].network.actor.state_dict(), model_name[0]+'.pth') 
                torch.save(agents[0].network.critic.state_dict(), model_name[1]+'.pth')
                torch.save(agents[1].network.actor.state_dict(), model_name[2]+'.pth') 
                torch.save(agents[1].network.critic.state_dict(), model_name[3]+'.pth')
                break
        # Training complete, record scores and save model
        self.scores = scores
        
    # Display score results
    def display(self, x_axis, y_axis):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
        
    # Close the environment
    def terminate(self):
        self.env.close()
        print("NOTE: Environment closed. No further training can be done.")