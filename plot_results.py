import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

def simple_moving_avg( data, ticks=75 ):
    vals = []
    avg = deque( maxlen=ticks )
    for i, el in enumerate(data):
        avg.append( el )
        if i % ticks ==0 and i != 0: avg.popleft()
        vals.append( np.mean( avg ) )
    return vals


def plotResults( filepath ):
    data = pd.read_csv(filepath, names=["Reward", "Epsilon", "Run Time"])  
    reward = data["Reward"].to_numpy()
    epsilon = data["Epsilon"].to_numpy()
    episodes = np.array( range(len(reward)))
    sma_reward = simple_moving_avg( reward )

    # Create two subplots sharing y axis
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(episodes, reward, 'k-', linewidth=1)
    ax1.plot(episodes, sma_reward, 'r-', linewidth=2)
    ax1.set(title='Results from Training', ylabel='Reward')

    ax2.plot(episodes, epsilon, 'b-', linewidth=1)
    ax2.set(xlabel='Episode', ylabel='Epsilon')

    plt.show()


if __name__ == '__main__':
    plotResults( filepath = "rewards/oah33/DDQN2/20220423-170444/episode_1900.csv" )