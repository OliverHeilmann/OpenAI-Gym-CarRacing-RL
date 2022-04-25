# Plots results collected from training the RL algorithms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

def simple_moving_avg( data, ticks=75 ):
    """Calculate the simple moving average (of ticks) of input results."""
    vals = []
    avg = deque( maxlen=ticks )
    for i, el in enumerate(data):
        avg.append( el )
        if i % ticks ==0 and i != 0: avg.popleft()
        vals.append( np.mean( avg ) )
    return [vals, ticks]


def plotResults( filepaths ):
    """Plot results from csv files which were saved from training."""
    # Create two subplots sharing y axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    labels = []
    for tpath, mpath in filepaths:
        # create a meaningful legend
        labels.append( tpath.split("/")[2] )

        # read data and plot as SMA
        data12 = pd.read_csv(tpath, names=["Reward", "Epsilon", "Run Time"])  
        reward = data12["Reward"].to_numpy()
        epsilon = data12["Epsilon"].to_numpy()
        episodes = np.array( range(len(reward)))
        sma_reward, ticks = simple_moving_avg( reward )

        # ax1.plot(episodes, reward, 'k-', linewidth=1)
        ax1.plot(episodes, sma_reward, '-', linewidth=2)
        ax1.set(title=f'{ticks} Period Simple Moving Average of Training Reward Against Episode', ylabel='Reward')

        ax2.plot(episodes, epsilon, '-', linewidth=1)
        ax2.set(title=f'Epsilon Against Episode', ylabel='Epsilon')

        # now get data for plot 3s
        data3 = pd.read_csv(mpath, names=["Episode", "Avg Reward", "Epsilon", "Avg Time", "Max", "Min", "Std Dev"])  
        ax3.plot(data3["Episode"], data3["Avg Reward"], '-', linewidth=1)
        ax3.set(title=f'Testing Reward Against Episode', xlabel='Episode', ylabel='Avg Reward (10 Runs)')
    
    fig.tight_layout()  # add padding between figs
    plt.legend( labels )
    plt.show()


if __name__ == '__main__':

    # MUST BE SAME LENGTH!
    training_rewards = [   "rewards/oah33/DQN2/20220422-164216/episode_1200.csv",
                            # "rewards/oah33/DDQN1/20220422-190009/episode_300.csv",
                            "rewards/oah33/DDQN2/20220423-122311/episode_1900.csv",
                            "rewards/oah33/DDQN2/20220423-170444/episode_1900.csv",
                            "rewards/oah33/DDQN3_NN/20220424-140943/episode_1900.csv"
                        ]

    model_rewards = [   "episode_test_runs/oah33/20220425-170036/DQN2/episode_run_rewards.csv",
                        # "episode_test_runs/oah33/20220425-170036/DDQN1/episode_run_rewards.csv",
                        "episode_test_runs/oah33/20220425-170036/DDQN2_T1/episode_run_rewards.csv",
                        "episode_test_runs/oah33/20220425-170036/DDQN2_T2/episode_run_rewards.csv",
                        "episode_test_runs/oah33/20220425-170036/DDQN3_NN/episode_run_rewards.csv",
                    ]
    filepaths = [ [training_rewards[i], model_rewards[i]] for i in range(len(training_rewards)) ]
    plotResults( filepaths = filepaths )