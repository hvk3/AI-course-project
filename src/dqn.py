import gym
from gym import wrappers
import h5py
import sys

from agent import Agent, save_agent, load_agent

actual = 'SuperMarioBros-1-1-v0'
tiles = 'SuperMarioBros-1-1-Tiles-v0'

if (__name__ == '__main__'):
	agent_file = sys.argv[1]
	weights_file = sys.argv[2]
	use_CNN = (sys.argv[3] == 'use_CNN')
	use_DDQN = len(sys.argv) >= 5 and (sys.argv[4] == 'use_DDQN')
	ENV_NAME = tiles if not use_CNN else actual
	env = gym.make(ENV_NAME)
	# import pdb;pdb.set_trace()
	try:
		agent = load_agent(agent_file, weights_file)
	except:
		agent = Agent(env.observation_space.shape, use_CNN)
	while (agent.episodes_run <= 10000):
		prev_action, own_prev_action = None, False
		current_state = env.reset()
		if (use_CNN):
			current_state = agent.preprocess_input(current_state)
		agent.episodes_run += 1
		agent.own_moves = 0
		next_lvl, loss, move_num = 0, 0.0, 0
		own_moves = []
		while True:
			action, own_action = agent.get_action(env, current_state)
			agent.own_moves += 1. * own_action
			if (own_action):
				own_moves.append(agent.actions.index(action))
			move_num += 1
			next_state, reward, has_completed, info = env.step(action)
			if (use_CNN):
				next_state = agent.preprocess_input(next_state)
			if (not has_completed):
				modified_reward = agent.get_modified_reward(reward, has_completed, current_state, next_state, info)
			else:
				modified_reward = -10
			loss += modified_reward
			agent.save(current_state, next_state, modified_reward, action, has_completed)
			current_state = next_state
			if (has_completed):
				if (use_DDQN):
					agent.target_model.set_weights(agent.model.get_weights())
				print("episode: {}/{}, average reward: {}, own moves: {}/{}".format\
					(agent.episodes_run, 10000, loss * 1. / move_num, agent.own_moves, move_num))
				print 'Set of agent\'s own moves :', own_moves
				break
		if (agent.episodes_run % 4 == 0):
			if (len(agent.prev_actions) > 20000):
				agent.exp_replay(20000)
			save_agent(agent, agent_file, weights_file)
		env.env.change_level(next_lvl)
	env.close()
