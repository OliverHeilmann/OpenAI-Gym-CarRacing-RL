"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
from pyglet.window import key
import numpy as np

from utils import *
from agent_ddpg import *


# Show preview
def key_press(k, mod):
    if k == key.SPACE:
        global preview
        preview = True


def key_release(k, mod):
    if k == key.SPACE:
        global preview
        preview = False


# Parameters
num_episodes = 1000

gym.logger.set_level(40)
preview = False
best_result = 0
all_episode_reward = []

# Initialize simulation
env = gym.make('CarRacing-v0')
env.reset()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release

# Define custom standard deviation for noise
noise_std = np.array([0.1, 4 * 0.2], dtype=np.float32)
agent = AgentDDPG(env.action_space, model_outputs=2, noise_std=noise_std)

# Loop of episodes
for ep in range(num_episodes):
    state = env.reset()
    agent.reset()
    done = False
    episode_reward = 0
    out_of_track = 0

    # One-step-loop
    while not done:
        if preview:
            env.render()

        action, train_action = agent.get_action(state)

        # This will make steering much easier
        action /= 4
        new_state, reward, done, info = env.step(action)

        # Models action output has a different shape for this problem
        agent.learn(state, train_action, reward, new_state)
        state = new_state
        episode_reward += reward

        if reward < 0:
            out_of_track += 1
            if out_of_track > 200:
                break
        else:
            out_of_track = 0

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-100:]).mean()
    print('Episode ', ep, ' result:', episode_reward, '..last 100 Average results:', average_result)

    if episode_reward > best_result:
        print('Saving this model because it is the best one so far')
        agent.save_solution()
        best_result = episode_reward

episode_indices = [i+1 for i in range(num_episodes)]
plot_learning_curve(episode_indices, all_episode_reward, 'ddpg.png')