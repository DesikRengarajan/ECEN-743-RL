"""
ECEN 743: Reinforcement Learning
Policy Gradient Assignment
Code tested using
	1. gymnasium 0.27.1
	2. box2d-py  2.3.5
	3. pytorch   2.0.0
	4. Python    3.9.12
1 & 2 can be installed using pip install gymnasium[box2d]

General Instructions
1. This code consists of TODO blocks, read them carefully and complete each of the blocks
2. Type your code between the following lines
			###### TYPE YOUR CODE HERE ######
			#################################
3. The default hyperparameters should be able to solve LunarLander-v2 in the continuous setting
4. It is not necessary to modify the rest of the code for this assignment, feel free to do so if needed.

"""
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import math
from collections import deque

class value_network(nn.Module):
	'''
	Value Network: Designed to take in state as input and give value as output
	Used as a baseline in Policy Gradient (PG) algorithms
	'''
	def __init__(self,state_dim):
		'''
			state_dim (int): state dimenssion
		'''
		super(value_network, self).__init__()
		self.l1 = nn.Linear(state_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, 1)

	def forward(self,state):
		'''
		Input: State
		Output: Value of state
		'''
		v = F.tanh(self.l1(state))
		v = F.tanh(self.l2(v))
		return self.l3(v)


class policy_network(nn.Module):
	'''
	Policy Network: Designed for continous action space, where given a 
	state, the network outputs the mean and standard deviation of the action
	'''
	def __init__(self,state_dim,action_dim,log_std = 0.0):
		"""
			state_dim (int): state dimenssion
			action_dim (int): action dimenssion
			log_std (float): log of standard deviation (std)
		"""
		super(policy_network, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.l1 = nn.Linear(state_dim,64)
		self.l2 = nn.Linear(64,64)
		self.mean = nn.Linear(64,action_dim)
		self.log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	
	def forward(self,state):
		'''
		Input: State
		Output: Mean, log_std and std of action
		'''
		a = F.tanh(self.l1(state))
		a = F.tanh(self.l2(a))
		a_mean = self.mean(a)
		a_log_std = self.log_std.expand_as(a_mean)
		a_std = torch.exp(a_log_std)		
		return a_mean, a_log_std, a_std

	def select_action(self, state):
		'''
		Input: State
		Output: Sample drawn from a normal disribution with mean and std
		'''
		a_mean, _, a_std = self.forward(state)
		action = torch.normal(a_mean, a_std)
		return action
	
	def get_log_prob(self, state, action):
		'''
		Input: State, Action
		Output: log probabilities
		'''
		mean, log_std, std = self.forward(state)
		var = std.pow(2)
		log_density = -(action - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
		return log_density.sum(1, keepdim=True)


class PGAgent():
	'''
	An agent that performs different variants of the PG algorithm
	'''
	def __init__(self,
	 state_dim, 
	 action_dim,
	 discount=0.99,
	 lr=1e-3,
	 gpu_index=0,
	 seed=0,
	 env="LunarLander-v2"
	 ):
		"""
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			discount (float): discount factor
			lr (float): learning rate
			gpu_index (int): GPU used for training
			seed (int): Seed of simulation
			env (str): Name of environment
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.discount = discount
		self.lr = lr
		self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
		self.env_name = env
		self.seed = seed
		self.policy = policy_network(state_dim,action_dim)
		self.value = value_network(state_dim)
		self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
		self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=self.lr)

	def sample_traj(self,batch_size=2000,evaluate = False):
		'''
		Input: 
			batch_size: minimum batch size needed for update
			evaluate: flag to be set during evaluation
		Output:
			states, actions, rewards,not_dones, episodic reward 	
		'''
		self.policy.to("cpu") #Move network to CPU for sampling
		env = gym.make(args.env,continuous=True)
		states = []
		actions = []
		rewards = []
		n_dones = []
		curr_reward_list = []
		while len(states) < batch_size:
			state, _ = env.reset(seed=self.seed)
			curr_reward = 0
			for t in range(1000):
				state_ten = torch.from_numpy(state).float().unsqueeze(0)
				with torch.no_grad():
					if evaluate:
						action = self.policy(state_ten)[0][0].numpy() # Take mean action during evaluation
					else:
						action = self.policy.select_action(state_ten)[0].numpy() # Sample from distribution during training
				action = action.astype(np.float64)
				n_state,reward,terminated,truncated,_ = env.step(action) # Execute action in the environment
				done = terminated or truncated
				states.append(state)
				actions.append(action)
				rewards.append(reward)
				n_done = 0 if done else 1
				n_dones.append(n_done)
				state = n_state
				curr_reward += reward
				if done:
					break
			curr_reward_list.append(curr_reward)
		if evaluate:
			return np.mean(curr_reward_list)
		return states,actions,rewards,n_dones, np.mean(curr_reward_list)
	



	def update(self,states,actions,rewards,n_dones,update_type='Baseline'):
		'''
		TODO: Complete this block to update the policy using different variants of PG
		Inputs:
			states: list of states
			actions: list of actions
			rewards: list of rewards
			n_dones: list of not dones
			update_type: type of PG algorithm
		Output: 
			None
		'''
		self.policy.to(self.device) #Move policy to GPU
		if update_type == "Baseline":
			self.value.to(self.device)	#Move value to GPU
		states_ten = torch.from_numpy(np.stack(states)).to(self.device)   #Convert to tensor and move to GPU
		action_ten = torch.from_numpy(np.stack(actions)).to(self.device)  #Convert to tensor and move to GPU
		rewards_ten = torch.from_numpy(np.stack(rewards)).to(self.device) #Convert to tensor and move to GPU
		n_dones_ten = torch.from_numpy(np.stack(n_dones)).to(self.device) #Convert to tensor and move to GPU

		if update_type == "Rt":
			'''
			TODO: Peform PG using the cumulative discounted reward of the entire trajectory
			1. Compute the discounted reward of each trajectory (rt)
			2. rt should be of the same length as rewards_ten (refer to the homework for an exact formulation)
			3. Compute log probabilities using states_ten and action_ten
			4. Compute policy loss and update the policy
			'''
			rt = torch.zeros(rewards_ten.shape[0],1).to(self.device)
			###### TYPE YOUR CODE HERE ######
			# Do steps 1-4
			################################# 

		if update_type == 'Gt':
			'''
			TODO: Peform PG using reward_to_go
			1. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
			2. gt should be of the same length as rewards_ten
			3. Compute log probabilities using states_ten and action_ten
			4. Compute policy loss and update the policy
			'''
			gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)

			###### TYPE YOUR CODE HERE ######
			# Compute reward_to_go (gt) 
			#################################

			gt = (gt - gt.mean()) / gt.std() #Helps with learning stablity

			###### TYPE YOUR CODE HERE ######
			# Compute log probabilities and update the policy
			#################################

		if update_type == 'Baseline':
			'''
			TODO: Peform PG using reward_to_go and baseline
			1. Compute values of states, this will be used as the baseline 
			2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
			3. gt should be of the same length as rewards_ten
			4. Compute advantages 
			5. Update the value network to predict gt for each state (L2 norm)
			6. Compute log probabilities using states_ten and action_ten
			7. Compute policy loss (using advantages) and update the policy
			'''
			with torch.no_grad():
				values_adv = self.value(states_ten)
			gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)

			###### TYPE YOUR CODE HERE ######
			# Compute reward_to_go (gt) and advantages	
			#################################

			advantages = (advantages - advantages.mean()) / advantages.std()

			###### TYPE YOUR CODE HERE ######
			# Do steps 5-7
			#################################



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="LunarLander-v2")			 # Gymnasium environment name
	parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--n-iter", default=200, type=int)           # Maximum number of training iterations
	parser.add_argument("--discount", default=0.99)                  # Discount factor
	parser.add_argument("--batch-size", default=5000, type=int)      # Training samples in each batch of training
	parser.add_argument("--lr", default=5e-3,type=float)             # Learning rate
	parser.add_argument("--gpu-index", default=0,type=int)		     # GPU index
	parser.add_argument("--algo", default="Baseline",type=str)		 # PG algorithm type. Baseline/Gt/Rt
	args = parser.parse_args()

	# Making the environment	
	env = gym.make(args.env,continuous=True)

	# Setting seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	kwargs = {
		"state_dim":state_dim,
		"action_dim":action_dim,
		"discount":args.discount,
		"lr":args.lr,
		"gpu_index":args.gpu_index,
		"seed":args.seed,
		"env":args.env
	}	
	learner = PGAgent(**kwargs) # Creating the PG learning agent

	moving_window = deque(maxlen=10)

	for e in range(args.n_iter):
		'''
		Steps of PG algorithm
			1. Sample environment to gather data using a policy
			2. Update the policy using the data
			3. Evaluate the updated policy
			4. Repeat 1-3
		'''
		states,actions,rewards,n_dones,train_reward = learner.sample_traj(batch_size=args.batch_size)
		learner.update(states,actions,rewards,n_dones,args.algo)
		eval_reward= learner.sample_traj(evaluate=True)
		moving_window.append(eval_reward)
		print('Training Iteration {} Training Reward: {:.2f} Evaluation Reward: {:.2f} \
		Average Evaluation Reward: {:.2f}'.format(e,train_reward,eval_reward,np.mean(moving_window)))
		
		"""
		TODO: Write code for
		1. Logging and plotting
		2. Rendering the trained agent 
		"""
		###### TYPE YOUR CODE HERE ######
		#################################

