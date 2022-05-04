"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
from pyglet.window import key
import numpy as np
import time
from utils import *
from agent_ddpg import *
import datetime

num_episodes = 50

gym.logger.set_level(40)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TEST_REWARD_DIR = f"test rewards/{TIMESTAMP}/"

avg_runs = []
for i in range(0, 1500, 100):
    env = gym.make('CarRacing-v0')
    env.reset()

    agent = AgentDDPG(env.action_space, model_outputs=2)
    print('loading ', 'models/'+ str(i) + '_')
    agent.load_model(path='models/', num=str(i) + '_', compiles=True)
    all_episode_reward = []
    run_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        agent.reset()
        done = False
        episode_reward = 0
        out_of_track = 0
        start_time = time.time()
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
                if out_of_track > 200:
                    break
            else:
                out_of_track = 0

        time_length = time.time() - start_time
        all_episode_reward.append(episode_reward)
        average_result = np.array(all_episode_reward[-100:]).mean()
        run_rewards.append([episode_reward, np.nan, time_length, np.nan, np.nan, np.nan])
        print('training episodes: ', i, 'run: ', ep, ' result:', episode_reward, '..last 100 Average results:', average_result,
              'Time:', "%0.2fs." % time_length)

    rr = [i[0] for i in run_rewards]
    rt = [i[2] for i in run_rewards]

    r_max = max(rr)
    r_min = min(rr)
    r_std_dev = np.std(rr)
    r_avg = np.mean(rr)
    t_avg = np.mean(rt)

    avg_runs.append([i, r_avg, np.nan, t_avg, r_max, r_min, r_std_dev])
    save_result_to_csv(f"run2", avg_runs, TEST_REWARD_DIR)
    print(f"[INFO]: Runs {num_episodes} | Avg Run Reward: ", "%0.2f" % r_avg, "| Avg Time:", "%0.2fs" % t_avg,
          f" | Max: {r_max} | Min: {r_min} | Std Dev: {r_std_dev}")


