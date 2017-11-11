import gym
from gym import wrappers
import sys

from naive_agent import Agent, save_agent, load_agent

ENV_NAME = 'SuperMarioBros-1-1-Tiles-v0'

if (__name__ == '__main__'):
	env = gym.make(ENV_NAME)
	agent_file = sys.argv[1]
	weights_file = sys.argv[2]
	use_CNN = (sys.argv[3] == 'use_CNN')
	try:
		agent = load_agent(agent_file, weights_file)
	except:
		agent = Agent(env.observation_space.shape, use_CNN)
	while (agent.episodes_run <= 10000):
		current_state = env.reset()
		agent.own_moves = 0
		agent.episodes_run += 1
		next_lvl, loss, total_moves = 0, 0.0, 0
		while True:
			action, own_action = agent.get_action(env, current_state)
			agent.own_moves += 1. * own_action
			total_moves += 1
			next_state, reward, has_completed, info = env.step(action)
			modified_reward = agent.get_modified_reward(reward, has_completed, current_state, next_state, info)
			loss += modified_reward
			agent.save(current_state, next_state, modified_reward, action, has_completed)
			current_state = next_state
			if (has_completed):
				print("episode: {}/{}, final reward: {}, own moves: {}/{}".format(agent.episodes_run, 10000, loss, agent.own_moves, total_moves))
				break
		if (len(agent.prev_actions) > 10000):
			agent.exp_replay(10000)
		env.env._reset_info_vars()
		save_agent(agent, agent_file, weights_file)
		env.env.change_level(next_lvl)
	env.close()
