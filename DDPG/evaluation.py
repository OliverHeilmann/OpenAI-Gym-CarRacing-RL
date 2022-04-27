"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
from pyglet.window import key
import numpy as np

from utils import *
from agent_ddpg import *


num_episodes = 500

gym.logger.set_level(40)
all_episode_reward = []

env = gym.make('CarRacing-v0')
env.reset()

noise_mean = np.array([0.0, -0.83], dtype=np.float32)
noise_std = np.array([0.0, 4 * 0.02], dtype=np.float32)
agent = AgentDDPG(env.action_space, model_outputs=2, noise_mean=noise_mean, noise_std=noise_std)
agent.load_solution('models/')


for ep in range(num_episodes):
    state = env.reset()
    agent.reset()
    done = False
    episode_reward = 0
    out_of_track = 0

    while not done:
        env.render()

        action, train_action = agent.get_action(state, add_noise=False)

        # This will make steering much easier
        action /= 4
        new_state, reward, done, info = env.step(action)

        state = new_state
        episode_reward += reward

        if reward < 0:
            out_of_track += 1
            if out_of_track > 150:
                break
        else:
            out_of_track = 0

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-100:]).mean()
    print('Episode ', ep, ' result:', episode_reward, '..last 100 Average results:', average_result)

