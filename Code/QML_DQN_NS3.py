# QML as Q Learning function approximator
# Need to specify STATE input format
# Computational Basis Encoding
# action output is still softmax [a_0, a_1, a_2, a_3, a_4, a_5]
# Deep Q-Learning DQN
# Experimence Replay (For i.i.d sampling) 
# Target Network (Updata every C episodes) ==> Another Circuit Parameter Set

# This version is enhanced with PyTorch and CUDA
# Should change the variable and optimizer to Torch Scheme
# Environment: OpenAI gym Cliff_Walking
# https://github.com/dennybritz/reinforcement-learning/tree/master/lib/envs

##
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import torch
import torch.nn as nn 
from torch.autograd import Variable

# from sklearn.datasets import make_circles
# from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from collections import deque

import gym
import time
import random
from collections import namedtuple
from copy import deepcopy

from NS3_Huck import Radio



Transition = namedtuple('Transition',
						('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def output_all(self):
		return self.memory

	def __len__(self):
		return len(self.memory)
####

## Saving Functions ##

def save_all_the_current_info(exp_name, file_title, iter_count, var_Q_circuit, var_Q_bias, iter_reward):
	## Saving the model
	file_title = exp_name + "/" + file_title + "_Iter_Count_" + str(iter_count)
	with open(file_title + "_var_Q_circuit" + ".txt", "wb") as fp:
			pickle.dump(var_Q_circuit, fp)

	with open(file_title + "_var_Q_bias" + ".txt", "wb") as fp:
			pickle.dump(var_Q_bias, fp)

	with open(file_title + "_iter_reward" + ".txt", "wb") as fp:
			pickle.dump(iter_reward, fp)
	if iter_count + 1 > 19: 
		full_plotting(file_title, iter_count + 1, iter_reward)


''' 
## Plotting Function 
Note: the plotting code is origin from Yang, Chao-Han Huck, et al. "Enhanced Adversarial Strategically-Timed Attacks Against Deep Reinforcement Learning." 
## ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE, 2020.
If you use the code in your research, please cite the original reference. 
'''

def full_plotting(_fileTitle, _trainingLength, _currentRewardList):
	scores = []                        # list containing scores from each episode
	scores_std = []                    # List containing the std dev of the last 100 episodes
	scores_avg = []                    # List containing the mean of the last 100 episodes
	scores_window = deque(maxlen=100)  # last 100 scores

	reward_list = _currentRewardList
	for i_episode in range(_trainingLength):
		score = reward_list[i_episode]

		scores_window.append(score)       # save most recent score
		scores.append(score)              # save most recent score
		scores_std.append(np.std(scores_window)) # save most recent std dev
		scores_avg.append(np.mean(scores_window)) # save most recent std dev

	na_raw = np.array(scores)
	na_mu = np.array(scores_avg)
	na_sigma = np.array(scores_std)

	# plot the scores
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

	# plot the sores by episode
	ax1.plot(np.arange(len(na_raw)), na_raw, label='Raw Score')
	leg1 = ax1.legend();
	ax1.set_xlim(0, len(na_raw)+1)
	ax1.set_ylabel('Score')
	ax1.set_xlabel('Episode #')
	ax1.set_title('raw scores')

	# plot the average of these scores
	ax2.axhline(y=100., xmin=0.0, xmax=1.0, color='r', linestyle='--', linewidth=0.7, alpha=0.9, label = 'Solved')
	ax2.plot(np.arange(len(na_mu)), na_mu, label='Average Score')
	leg2 = ax2.legend()
	ax2.fill_between(np.arange(len(na_mu)), na_mu+na_sigma, na_mu-na_sigma, facecolor='gray', alpha=0.1)
	ax2.set_ylabel('Average Score')
	ax2.set_xlabel('Episode #')
	ax2.set_title('average scores')

	f.tight_layout()

	# f.savefig(fig_prefix + 'dqn.eps', format='eps', dpi=1200)
	f.savefig(_fileTitle + '_full.pdf',format = 'pdf')

def plotTrainingResultCombined(_iter_index, _iter_reward, _iter_total_steps, _fileTitle):
	fig, ax = plt.subplots()
	# plt.yscale('log')
	ax.plot(_iter_index, _iter_reward, '-b', label='Reward')
	ax.plot(_iter_index, _iter_total_steps, '-r', label='Total Steps')
	leg = ax.legend();

	ax.set(xlabel='Iteration Index', 
		   title=_fileTitle)
	fig.savefig(_fileTitle + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".png")

def plotTrainingResultReward(_iter_index, _iter_reward, _iter_total_steps, _fileTitle):
	fig, ax = plt.subplots()
	# plt.yscale('log')
	ax.plot(_iter_index, _iter_reward, '-b', label='Reward')
	# ax.plot(_iter_index, _iter_total_steps, '-r', label='Total Steps')
	leg = ax.legend();

	ax.set(xlabel='Iteration Index', 
		   title=_fileTitle)
	fig.savefig(_fileTitle + "_REWARD" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".png")


########################################

def decimalToBinaryFixLength(_length, _decimal):
	binNum = bin(int(_decimal))[2:]
	outputNum = [int(item) for item in binNum]
	if len(outputNum) < _length:
		outputNum = np.concatenate((np.zeros((_length-len(outputNum),)),np.array(outputNum)))
	else:
		outputNum = np.array(outputNum)
	return outputNum

def NS_3_quantum_state(state_tuple):
	channel_num = len(state_tuple[0])
	state = np.array(state_tuple[0])
	state = state.argmax()
	env_time = state_tuple[1]
	state_ind = state * channel_num + env_time%channel_num
	
	return decimalToBinaryFixLength(channel_num, state_ind)


## PennyLane Part ##

# Specify the datatype of the Totch tensor
dtype = torch.DoubleTensor

## Define a FOUR qubit system
dev = qml.device('default.qubit', wires=4)
# dev = qml.device('qiskit.basicaer', wires=4)
def statepreparation(a):

	"""Quantum circuit to encode a the input vector into variational params

	Args:
		a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
	"""
	
	# Rot to computational basis encoding
	# a = [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8]

	for ind in range(len(a)):
		qml.RX(np.pi * a[ind], wires=ind)
		qml.RZ(np.pi * a[ind], wires=ind)


def layer(W):
	""" Single layer of the variational classifier.

	Args:
		W (array[float]): 2-d array of variables for one layer
	"""

	qml.CNOT(wires=[0, 1])
	qml.CNOT(wires=[1, 2])
	qml.CNOT(wires=[2, 3])
	# qml.CNOT(wires=[3, 4])
	# qml.CNOT(wires=[4, 5])


	qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
	qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
	qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
	qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)
	# qml.Rot(W[4, 0], W[4, 1], W[4, 2], wires=4)
	# qml.Rot(W[5, 0], W[5, 1], W[5, 2], wires=5)

	
	

@qml.qnode(dev, interface='torch')
def circuit(weights, angles=None):
	"""The circuit of the variational classifier."""
	# Can consider different expectation value
	# PauliX , PauliY , PauliZ , Identity  

	statepreparation(angles)
	
	for W in weights:
		layer(W)

	return [qml.expval(qml.PauliZ(ind)) for ind in range(4)]


def variational_classifier(var_Q_circuit, var_Q_bias , angles=None):
	"""The variational classifier."""

	# Change to SoftMax???

	weights = var_Q_circuit
	# bias_1 = var_Q_bias[0]
	# bias_2 = var_Q_bias[1]
	# bias_3 = var_Q_bias[2]
	# bias_4 = var_Q_bias[3]
	# bias_5 = var_Q_bias[4]
	# bias_6 = var_Q_bias[5]

	# raw_output = circuit(weights, angles=angles) + np.array([bias_1,bias_2,bias_3,bias_4,bias_5,bias_6])
	raw_output = circuit(weights, angles=angles) + var_Q_bias
	# We are approximating Q Value
	# Maybe softmax is no need
	# softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()

	return raw_output


def square_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	loss = 0
	for l, p in zip(labels, predictions):
	    loss = loss + (l - p) ** 2
	loss = loss / len(labels)
	# print("LOSS")

	# print(loss)

	# output = torch.abs(predictions - labels)**2
	# output = torch.sum(output) / len(labels)

	# loss = nn.MSELoss()
	# output = loss(labels.double(), predictions.double())

	return loss

# def square_loss(labels, predictions):
# 	""" Square loss function

# 	Args:
# 		labels (array[float]): 1-d array of labels
# 		predictions (array[float]): 1-d array of predictions
# 	Returns:
# 		float: square loss
# 	"""
# 	# In Deep Q Learning
# 	# labels = target_action_value_Q
# 	# predictions = action_value_Q

# 	# loss = 0
# 	# for l, p in zip(labels, predictions):
# 	# 	loss = loss + (l - p) ** 2
# 	# loss = loss / len(labels)

# 	# loss = nn.MSELoss()
# 	output = torch.abs(predictions - labels)**2
# 	output = torch.sum(output) / len(labels)
# 	# output = loss(torch.tensor(predictions), torch.tensor(labels))
# 	# print("LOSS OUTPUT")
# 	# print(output)

# 	return output

def abs_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	# In Deep Q Learning
	# labels = target_action_value_Q
	# predictions = action_value_Q

	# loss = 0
	# for l, p in zip(labels, predictions):
	# 	loss = loss + (l - p) ** 2
	# loss = loss / len(labels)

	# loss = nn.MSELoss()
	output = torch.abs(predictions - labels)
	output = torch.sum(output) / len(labels)
	# output = loss(torch.tensor(predictions), torch.tensor(labels))
	# print("LOSS OUTPUT")
	# print(output)

	return output

def huber_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	# In Deep Q Learning
	# labels = target_action_value_Q
	# predictions = action_value_Q

	# loss = 0
	# for l, p in zip(labels, predictions):
	# 	loss = loss + (l - p) ** 2
	# loss = loss / len(labels)

	# loss = nn.MSELoss()
	loss = nn.SmoothL1Loss()
	# output = loss(torch.tensor(predictions), torch.tensor(labels))
	# print("LOSS OUTPUT")
	# print(output)

	return loss(labels, predictions)


def cost(var_Q_circuit, var_Q_bias, features, labels):
	"""Cost (error) function to be minimized."""

	# predictions = [variational_classifier(weights, angles=f) for f in features]
	# Torch data type??
	
	predictions = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles= NS_3_quantum_state(item.state))[item.action] for item in features]
	# predictions = torch.tensor(predictions,requires_grad=True)
	# labels = torch.tensor(labels)
	# print("PRIDICTIONS:")
	# print(predictions)
	# print("LABELS:")
	# print(labels)

	return square_loss(labels, predictions)


#############################

def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, s, train=False):
	"""
	@param Q Q values state x action -> value
	@param epsilon for exploration
	@param s number of states
	@param train if true then no random actions selected
	"""

	# Modify to incorporate with Variational Quantum Classifier
	# epsilon should change along training
	# In the beginning => More Exploration
	# In the end => More Exploitation

	# More Random
	np.random.seed(int(datetime.now().strftime("%S%f")))


	if train or np.random.rand() < ((epsilon/n_actions)+(1-epsilon)):
		# action = np.argmax(Q[s, :])
		# variational classifier output is torch tensor
		# action = np.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(9,s)))
		action = torch.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = NS_3_quantum_state(s)))
	else:
		# need to be torch tensor
		action = torch.tensor(np.random.randint(0, n_actions))
	return action



def deep_Q_Learning(exp_name, file_title, alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
	"""
	@param alpha learning rate
	@param gamma decay factor
	@param epsilon for exploration
	@param max_steps for max step in each episode
	@param n_tests number of test episodes
	"""

	
	env = Radio()
	n_actions = 4
	# env = gym.make('Deterministic-4x4-FrozenLake-v0')
	# n_states, n_actions = env.observation_space.n, env.action_space.n
	# print("NUMBER OF STATES:" + str(n_states))
	# print("NUMBER OF ACTIONS:" + str(n_actions))

	# Initialize Q function approximator variational quantum circuit
	# initialize weight layers

	num_qubits = 4
	num_layers = 2
	# var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	var_init_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True)
	var_init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True)

	# Define the two Q value function initial parameters
	# Use np copy() function to DEEP COPY the numpy array
	var_Q_circuit = var_init_circuit
	var_Q_bias = var_init_bias
	# print("INIT PARAMS")
	# print(var_Q_circuit)

	var_target_Q_circuit = var_Q_circuit.clone().detach()
	var_target_Q_bias = var_Q_bias.clone().detach()

	##########################
	# Optimization method => random select train batch from replay memory
	# and opt

	# opt = NesterovMomentumOptimizer(0.01)

	# opt = torch.optim.Adam([var_Q_circuit, var_Q_bias], lr = 0.1)
	# opt = torch.optim.SGD([var_Q_circuit, var_Q_bias], lr=0.1, momentum=0.9)
	opt = torch.optim.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

	## NEed to move out of the function
	TARGET_UPDATE = 20
	batch_size = 5
	OPTIMIZE_STEPS = 5
	##


	target_update_counter = 0

	iter_index = []
	iter_reward = []
	iter_total_steps = []

	cost_list = []


	timestep_reward = []


	# Demo of generating a ACTION
	# Output a numpy array of value for each action

	# Define the replay memory
	# Each transition:
	# (s_t_0, a_t_0, r_t, s_t_1, 'DONE')

	memory = ReplayMemory(1000)

	# Input Angle = decimalToBinaryFixLength(9, stateInd)
	# Input Angle is a numpy array

	# stateVector = decimalToBinaryFixLength(9, stateInd)

	# q_val_s_t = variational_classifier(var_Q, angles=stateVector)
	# # action_t = q_val_s_t.argmax()
	# action_t = epsilon_greedy(var_Q, epsilon, n_actions, s)
	# q_val_target_s_t = variational_classifier(var_target_Q, angles=stateVector)

	# train the variational classifier


	for episode in range(episodes):
		print(f"Episode: {episode}")
		# Output a s in decimal format
		# s = env.reset()

		env = Radio()
		s = ([0,0,0,0], 0)

		# Doing epsilog greedy action selection
		# With var_Q
		a = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s).item()
		t = 0
		total_reward = 0
		done = False


		while t < max_steps:
			if render:
				print("###RENDER###")
				env.render()
				print("###RENDER###")
			t += 1

			target_update_counter += 1

			# Execute the action 
			s_, reward, done = env.step(a)
			env_time = env.t
			s_ = np.array(s_)

			s_ = (s_, env_time)

			# print("Reward : " + str(reward))
			# print("Done : " + str(done))
			total_reward += reward
			# a_ = np.argmax(Q[s_, :])
			a_ = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s_).item()
			
			# print("ACTION:")
			# print(a_)

			memory.push(s, a, reward, s_, done)

			if len(memory) > batch_size:

				# Sampling Mini_Batch from Replay Memory

				batch_sampled = memory.sample(batch_size = batch_size)

				# Transition = (s_t, a_t, r_t, s_t+1, done(True / False))

				# item.state => state
				# item.action => action taken at state s
				# item.reward => reward given based on (s,a)
				# item.next_state => state arrived based on (s,a)

				Q_target = [item.reward + (1 - int(item.done)) * gamma * torch.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles= NS_3_quantum_state(item.next_state))) for item in batch_sampled]
				# Q_prediction = [variational_classifier(var_Q, angles=decimalToBinaryFixLength(9,item.state))[item.action] for item in batch_sampled ]

				# Gradient Descent
				# cost(weights, features, labels)
				# square_loss_training = square_loss(labels = Q_target, Q_predictions)
				# print("UPDATING PARAMS...")

				# CHANGE TO TORCH OPTIMIZER
				
				# var_Q = opt.step(lambda v: cost(v, batch_sampled, Q_target), var_Q)
				# opt.zero_grad()
				# loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
				# print(loss)
				# FIX this gradient error
				# loss.backward()
				# opt.step(loss)

				def closure():
					opt.zero_grad()
					loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
					# print(loss)
					loss.backward()
					return loss
				opt.step(closure)

				# print("UPDATING PARAMS COMPLETED")
				# current_replay_memory = memory.output_all()
				# current_target_for_replay_memory = [item.reward + (1 - int(item.done)) * gamma * torch.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles=decimalToBinaryFixLength(9,item.next_state))) for item in current_replay_memory]
				# current_target_for_replay_memory = [item.reward + (1 - int(item.done)) * gamma * np.max(variational_classifier(var_target_Q, angles=decimalToBinaryFixLength(9,item.next_state))) for item in current_replay_memory]

				# if t%5 == 0:
				# 	cost_ = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = current_replay_memory, labels = current_target_for_replay_memory)
				# 	print("Cost: ")
				# 	print(cost_.item())
				# 	cost_list.append(cost_)


			if target_update_counter > TARGET_UPDATE:
				print("UPDATEING TARGET CIRCUIT...")

				var_target_Q_circuit = var_Q_circuit.clone().detach()
				var_target_Q_bias = var_Q_bias.clone().detach()
				
				target_update_counter = 0

			s, a = s_, a_

			epsilon = epsilon * 0.99

			if done:
				if render:
					print("###FINAL RENDER###")
					env.render()
					print("###FINAL RENDER###")
					print(f"This episode took {t} timesteps and reward: {total_reward}")
				# epsilon = epsilon / ((episode/10.) + 1)
				# print("Q Circuit Params:")
				# print(var_Q_circuit)
				print(f"This episode took {t} timesteps and reward: {total_reward}")
				timestep_reward.append(total_reward)
				iter_index.append(episode)
				iter_reward.append(total_reward)
				iter_total_steps.append(t)
				save_all_the_current_info(exp_name, file_title, episode, var_Q_circuit, var_Q_bias, iter_reward)
				break
	# if render:
	# 	print(f"Here are the Q values:\n{Q}\nTesting now:")
	# if test:
	# 	test_agent(Q, env, n_tests, n_actions)
	return timestep_reward, iter_index, iter_reward, iter_total_steps, var_Q_circuit, var_Q_bias


# def test_agent(Q, env, n_tests, n_actions, delay=1):
# 	for test in range(n_tests):
# 		print(f"Test #{test}")
# 		s = env.reset()
# 		done = False
# 		epsilon = 0
# 		while True:
# 			time.sleep(delay)
# 			env.render()
# 			a = epsilon_greedy(Q, epsilon, n_actions, s, train=True)
# 			print(f"Chose action {a} for state {s}")
# 			s, reward, done, info = env.step(a)
# 			if done:
# 				if reward > 0:
# 					print("Reached goal!")
# 				else:
# 					print("Shit! dead x_x")
# 				time.sleep(3)
# 				break

# Should add plotting function and KeyboardInterrupt Handler

if __name__ =="__main__":
	alpha = 0.4
	gamma = 0.999
	epsilon = 1.
	episodes = 500
	max_steps = 100
	n_tests = 2

	file_title = 'VQDQN_NS3_4_Channels_RMSProp' + datetime.now().strftime("NO%Y%m%d%H%M%S")
	exp_name = 'VQDQN_NS3_4_Channels_Exp_2'

	import os

	if not os.path.exists(exp_name):
	    os.makedirs(exp_name)

	timestep_reward, iter_index, iter_reward, iter_total_steps , var_Q_circuit, var_Q_bias = deep_Q_Learning(exp_name, file_title, alpha, gamma, epsilon, episodes, max_steps, n_tests, test = False)
	
	print(timestep_reward)
	

	## Drawing Training Result ##
	
	
	plotTrainingResultReward(_iter_index = iter_index, _iter_reward = iter_reward, _iter_total_steps = iter_total_steps, _fileTitle = file_title)

	## Saving the model
	with open(file_title + "_var_Q_circuit" + ".txt", "wb") as fp:
			pickle.dump(var_Q_circuit, fp)

	with open(file_title + "_var_Q_bias" + ".txt", "wb") as fp:
			pickle.dump(var_Q_bias, fp)

	with open(file_title + "_iter_reward" + ".txt", "wb") as fp:
			pickle.dump(iter_reward, fp)










