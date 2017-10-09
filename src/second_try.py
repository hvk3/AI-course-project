# Reference : https://keon.io/deep-q-learning/

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adadelta, Adam

import gym
from gym import wrappers
import numpy as np
import random
import time

ENV_NAME = 'SuperMarioBros-1-1-Tiles-v0'

class Agent:
	def __init__(self, input_shape):
		self.input_shape = input_shape
		self.discount_factor = 0.95
		self.prev_actions = []
		self.actions = self.get_actions()
		self.nb_actions = len(self.actions)
		self.learning_rate = 0.001
		self.model = self.get_model(self.input_shape, self.nb_actions)
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

	def get_model(self, input_shape, nb_actions):
		model = Sequential()
		model.add(Flatten(input_shape = input_shape))
		model.add(Dense(input_dim = model.layers[-1].output_shape, units = 24, activation = 'relu'))
		model.add(Dense(input_dim = model.layers[-1].output_shape, units = 24, activation = 'relu'))
		model.add(Dense(input_dim = model.layers[-1].output_shape, units = nb_actions, activation = 'linear'))
		model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
		return model

	def get_actions(self):
		A = [0, 0, 0, 0, 1, 0]
		# B = [0, 0, 0, 0, 0, 1]
		left = [0, 1, 0, 0, 0, 0]
		left_A = [0, 1, 0, 0, 1, 0]
		# left_B = [0, 1, 0, 0, 0, 1]
		right = [0, 0, 0, 1, 0, 0]
		right_A = [0, 0, 0, 1, 1, 0]
		# right_B = [0, 0, 0, 1, 0, 1]
		# actions = [A, B, left, left_A, left_B, right, right_A, right_B]
		actions = [A, left, left_A, right, right_A]
		return actions

	def save(self, current_state, next_state, reward, action, has_completed):
		self.prev_actions.append((current_state, next_state, reward, action, has_completed))

	def get_action(self, current_state):
		if (np.random.rand() <= self.epsilon):
			return random.randrange(self.nb_actions)
		best_action = np.argmax(self.model.predict(np.array([current_state]))[0])
		return best_action

	def exp_replay(self, batch_size):
		batch = random.sample(self.prev_actions, batch_size)
		for (current_state, next_state, reward, action, has_completed) in batch:
			target = reward
			if (has_completed):
				target = reward + self.discount_factor * np.max(self.model.predict(np.array([current_state]))[0])
			target_f = self.model.predict(np.array([current_state]))
			target_f[0][action] = target
			self.model.fit(np.array([current_state]), target_f, epochs = 1, verbose = 0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def save_weights(self, filename):
		self.model.save_weights(filename)
	
	def load_weights(self, filename):
		self.model.load_weights(filename)

if (__name__ == '__main__'):
	env = gym.make(ENV_NAME)
	agent = Agent(env.observation_space.shape)
	try:
		agent.load_weights('first_attempt.h5')
	except:
		pass
	env.close()
	num_episodes = 2000
	for e in xrange(1, num_episodes + 1):
		env = gym.make(ENV_NAME)
		current_state = env.reset()
		while True:
			action = agent.get_action(current_state)
			next_state, reward, has_completed, info = env.step(agent.actions[action])
			agent.save(current_state, next_state, reward, action, has_completed)
			current_state = next_state
			if (has_completed):
				print("episode: {}/{}, score: {}".format(e, num_episodes, info["total_reward"]))
				break
		if (len(agent.prev_actions) > 2000):
			agent.prev_actions = agent.prev_actions[-2000:]
		if (len(agent.prev_actions) > 16):
			agent.exp_replay(16)
		env.close()
		if (e % 5 == 0):
			agent.save_weights('first_attempt.h5')