# Reference for CNN : https://keon.io/deep-q-learning/
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization

import cv2
from collections import deque
import numpy as np
import os
import pickle
import random

np.random.seed(0)

def save_agent(agent, agent_filename, weights_filename):
	agent.model.save_weights(weights_filename)
	agent.model = None
	with open(agent_filename, 'w') as f:
		pickle.dump(agent, f)
	if (agent.use_CNN):
		agent.model = agent.get_CNN(agent.input_shape, agent.nb_actions)
	else:
		agent.model = agent.get_NN(agent.input_shape, agent.nb_actions)
	agent.model.load_weights(weights_filename)

def load_agent(agent_filename, weights_filename):
	with open(agent_filename, 'r') as f:
		agent = pickle.load(f)
	if (agent.use_CNN):
		agent.model = agent.get_CNN(agent.input_shape, agent.nb_actions)
	else:
		agent.model = agent.get_NN(agent.input_shape, agent.nb_actions)
	agent.model.load_weights(weights_filename)
	return agent

class Agent:
	def __init__(self, input_shape, use_CNN = False, use_DDQN = False):
		self.input_shape = input_shape
		self.matrix_visualisation = {'empty' : 0, 'object' : 1, 'enemies' : 2, 'mario' : 3}
		self.use_CNN = use_CNN
		self.use_DDQN = use_DDQN
		self.discount_factor = 0.95
		self.prev_actions = deque(maxlen = 200000)
		self.own_moves = 0
		self.actions = self.get_actions()
		self.nb_actions = len(self.actions)
		self.learning_rate = 0.001
		if (use_CNN):
			self.model = self.get_CNN(self.input_shape, self.nb_actions)
			if (use_DDQN):
				self.target_model = self.get_CNN(self.input_shape, self.nb_actions)
				self.target_model.set_weights(self.model.get_weights())
		else:
			self.model = self.get_NN(self.input_shape, self.nb_actions)
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.episodes_run = 0

	def get_NN(self, input_shape, nb_actions):
		model = Sequential()
		model.add(Flatten(input_shape = input_shape))
		model.add(Dense(128, activation = 'relu'))
		model.add(BatchNormalization())
		model.add(Dense(48, activation = 'relu'))
		model.add(Dense(nb_actions, activation = 'linear'))
		model.compile(loss = 'mse', optimizer = RMSprop(lr = self.learning_rate))
		return model

	def get_CNN(self, input_shape, nb_actions):
		model = Sequential()
		model.add(Conv2D(16, (3, 3), input_shape = input_shape, activation = 'relu',	data_format = 'channels_first'))
		model.add(MaxPooling2D(pool_size = (2, 2)))
		model.add(BatchNormalization())
		model.add(Conv2D(32, (3, 3), input_shape = model.layers[-1].output_shape, activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2, 2)))
		model.add(BatchNormalization())
		model.add(Flatten())
		model.add(Dense(nb_actions, activation = 'linear'))
		model.compile(loss = 'mse', optimizer = RMSprop(lr = self.learning_rate))
		return model

	def preprocess_input(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (84, 84))
		return img

	def get_actions(self):
		A = [0, 0, 0, 0, 1, 0]
		B = [0, 0, 0, 0, 0, 1]
		left = [0, 1, 0, 0, 0, 0]
		left_A = [0, 1, 0, 0, 1, 0]
		left_B = [0, 1, 0, 0, 0, 1]
		left_A_B = [0, 1, 0, 0, 1, 1]
		right = [0, 0, 0, 1, 0, 0]
		right_A = [0, 0, 0, 1, 1, 0]
		right_B = [0, 0, 0, 1, 0, 1]
		right_A_B = [0, 0, 0, 1, 1, 1]
		actions = [A, B, left, left_A, left_B, left_A_B, right, right_A, right_B, right_A_B]
		return actions

	def get_modified_reward(self, reward, has_completed, prev_state, current_state, info, max_score = 3200):
		unit_score = 100. / max_score
		modified_reward = reward * unit_score
		time_penalty_applied = False
		enemy_penalty_applied = False
		falling_penalty_applied = False
		if (info['distance'] > 40):
			if (len(np.where(current_state == self.matrix_visualisation['mario'])[0]) == 0):
				modified_reward -= 150
				falling_penalty_applied = True
		try:
			curr_y, curr_x = map(lambda x: x[0], np.where(current_state == self.matrix_visualisation['mario']))
			prev_y, prev_x = map(lambda x: x[0], np.where(prev_state == self.matrix_visualisation['mario']))
			if (prev_y > curr_y):
				modified_reward -= unit_score * (prev_y - curr_y)
			else:
				modified_reward += unit_score * 5 * (curr_y - prev_y)
			if (prev_y == curr_y):
				modified_reward -= unit_score
			if (has_completed):
				current_pos_mario_y, current_pos_mario_x = map(lambda x: list(x), np.where(current_state == self.matrix_visualisation['mario']))
				current_pos_enemies_y, current_pos_enemies_x = map(lambda x: list(x), np.where(current_state == self.matrix_visualisation['enemies']))
				for (mario_pos_y, mario_pos_x) in zip(current_pos_mario_y, current_pos_mario_x):
					for (enemy_pos_y, enemy_pos_x) in zip(current_pos_enemies_y, current_pos_enemies_x):
						if (np.abs(mario_pos_x - enemy_pos_x) <= 1 and np.abs(mario_pos_y - enemy_pos_y) <= 1):
							if (not enemy_penalty_applied):
								modified_reward -= 150
								enemy_penalty_applied = True
				if (not enemy_penalty_applied):
					if (not falling_penalty_applied):
						modified_reward -= 150
						time_penalty_applied = True
		except:
			pass
		return modified_reward

	def save(self, current_state, next_state, reward, action, has_completed):
		self.prev_actions.append((current_state, next_state, reward, action, has_completed))

	def get_prediction(self, state):
		return self.model.predict(np.array([state]))

	def get_action(self, env, current_state):
		if (self.use_CNN):
			temp_current_state = np.expand_dims(current_state, axis = 0)
		else:
			temp_current_state = current_state
		if (np.random.rand() <= self.epsilon):
			return self.actions[np.random.randint(low = 0, high = self.nb_actions)], False
		best_action = np.argmax(self.get_prediction(temp_current_state)[0])
		# print 'My best action:', best_action
		return self.actions[best_action], True

	def exp_replay(self, batch_size):
		batch = random.sample(self.prev_actions, batch_size)
		for (current_state, next_state, reward, action, has_completed) in batch:
			if (self.use_CNN):
				temp_current_state = np.expand_dims(current_state, axis = 0)
			else:
				temp_current_state = current_state
			target = reward
			if (not has_completed):
				target += self.discount_factor * np.max(self.get_prediction(temp_current_state)[0])
			target_f = self.get_prediction(temp_current_state)
			if (action in self.actions):
				target_f[0][self.actions.index(action)] = target
			else:
				continue
			self.model.fit(np.array([temp_current_state]), target_f, epochs = 1, verbose = 0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
