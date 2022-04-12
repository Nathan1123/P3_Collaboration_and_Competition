[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/Nathan1123/P3_Collaboration_and_Competition/main/result_chart.png "Results Chart"

# Project 3: Collaboration and Competition

Nathan Goedeke

Deep Reinforcement Learning

April 11, 2022

### Algorithm Summary

The DDPG model of reinforcement learning uses the Actor-Critic algorithm to solve this problem. The Actor-Critic method is designed to combine the low bias of Monte-Carlo method with the low variance of Temporal Difference. 

In this model, there are two neural networks in play: the Actor and the Critic. Just like in Deep Q learning, the Actor uses a deep neural network to estimate the optimal policy for a given problem. The Critic uses a deep neural network to estimate the cumulative episodic reward for each State-Action tuple of information (S,A,R,S',A').

This project uses two agents, where each agent has its own actor and its own critic, for a combined total of four neural networks. The state S is given two both agents independently. 

Conceptually, the agents are working in competition to score more points than their opponant. In reality, however, the agents are working collaboratively to keep the ball in air as long as possible, which benefits both of them. Thus, each agent sees the other agent as part of the environment, and factors that into its future policy.   

### Parameters

These are all the parameters used to generate this model. These first four parameters are already given by the nature of the problem:

* Score Threshold - minimal needed score before ending training (0.5)
* Seed - Pseudo-Random seed used in the learning model. Set to zero here for true randomness
* Score Window - Number of episodes to average for a total score (100)

These remaining parameters are determined by experimentation and research:

* Buffer size - Number of tuples (S,A,R,S',A') stored in the experience replay buffer
* Batch size - Size of batch data in the neural network
* Gamma - The discount factor, which determines how much information diminishes over multiple runs. A value of 1 remembers everything and a value of 0 remembers nothing.
* Weight Decay (Tau) - Controls how much information is balanced between the Target Q-Network and the Local Q-Network
* Learning Rate - How sensitive the neural network is to updating weights
* Update Every - How many actions are taken before updating the network weights
* Times update - How many updates to make each interval
* N Episodes - Total number of episodes to run
* Max T - Maximum number of actions to take before ending an episode. Otherwise, the episode will end when the environment returns Done=True

### Results

The agent successfully trained after 500 episodes to collect an average reward of 30 over 100 episodes. The results per 10 episodes and chart are displayed below. After 470 episodes, the minimal expected goal of 30 was eclipsed and training stopped. 

Episode 100	Average Score: 0.000 	Max score: 0.10000000149011612<br>
Episode 200	Average Score: 0.000 	Max score: 0.09000000171363354<br>
Episode 300	Average Score: 0.011 	Max score: 0.10000000149011612<br>
Episode 400	Average Score: 0.022 	Max score: 0.20000000298023224<br>
Episode 500	Average Score: 0.022 	Max score: 0.10000000149011612<br>
Episode 600	Average Score: 0.033 	Max score: 0.10000000149011612<br>
Episode 700	Average Score: 0.055 	Max score: 0.19000000320374966<br>
Episode 800	Average Score: 0.077 	Max score: 0.30000000447034836<br>
Episode 900	Average Score: 0.066 	Max score: 0.20000000298023224<br>
Episode 1000	Average Score: 0.088 	Max score: 0.4000000059604645<br>
Episode 1100	Average Score: 0.111 	Max score: 0.7000000104308128<br>
Episode 1200	Average Score: 0.133 	Max score: 0.5000000074505806<br>
Episode 1300	Average Score: 0.177 	Max score: 0.7000000104308128<br>
Episode 1400	Average Score: 0.188 	Max score: 0.7000000104308128<br>
Episode 1500	Average Score: 0.222 	Max score: 1.6000000238418581<br>
Episode 1600	Average Score: 0.255 	Max score: 1.0000000149011612<br>
Episode 1700	Average Score: 0.422 	Max score: 2.0000000298023224<br>
Episode 1726 	Average Score: 0.50 	Max score: 2.6000000387430194<br>
Environment solved in 1726 episodes!	Average Score: 0.50

![Results Chart][image1]

### Future Work

* Better performance could possibly be achieved by fine tuning other parameters. It seems likely the agent could get as high as 18
* Implementing a priority factor to the experience replay buffer could make better use of the replay system
* Exploring parallel processing of several agents for soccer