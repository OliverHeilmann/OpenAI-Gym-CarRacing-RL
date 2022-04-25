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
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    labels = []
    for filepath in filepaths:
        # create a meaningful legend
        labels.append( filepath.split("/")[2] )

        # read data and plot as SMA
        data = pd.read_csv(filepath, names=["Reward", "Epsilon", "Run Time"])  
        reward = data["Reward"].to_numpy()
        epsilon = data["Epsilon"].to_numpy()
        episodes = np.array( range(len(reward)))
        sma_reward, ticks = simple_moving_avg( reward )

        # ax1.plot(episodes, reward, 'k-', linewidth=1)
        ax1.plot(episodes, sma_reward, '-', linewidth=2)
        ax1.set(title=f'Simple Moving Average of {ticks} Ticks Showing Training Rewards Per Episode', ylabel='Reward')

        ax2.plot(episodes, epsilon, '-', linewidth=1)
        ax2.set(xlabel='Episode [unit]', ylabel='Epsilon [unit]')

    plt.legend( labels )
    plt.show()


if __name__ == '__main__':

    paths = [   "rewards/oah33/DQN2/20220422-164216/episode_1200.csv",
                "rewards/oah33/DDQN1/20220422-190009/episode_300.csv",
                "rewards/oah33/DDQN2/20220423-122311/episode_1900.csv",
                "rewards/oah33/DDQN2/20220423-170444/episode_1900.csv",
                "rewards/oah33/DDQN3_NN/20220424-140943/episode_1900.csv"
            ]

    plotResults( filepaths = paths)