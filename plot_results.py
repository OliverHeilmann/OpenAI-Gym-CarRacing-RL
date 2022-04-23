import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plotResults( filepath ):
    data = pd.read_csv(filepath, names=["Reward", "Epsilon"])  
    reward = data["Reward"].to_numpy()
    epsilon = data["Epsilon"].to_numpy()
    episodes = np.array( range(len(reward)))

    # Create two subplots sharing y axis
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(episodes, reward, 'k-')
    ax1.set(title='Results from Training', ylabel='Reward')

    ax2.plot(episodes, epsilon, 'r-')
    ax2.set(xlabel='Episode', ylabel='Epsilon')

    plt.show()


if __name__ == '__main__':
    plotResults( filepath = "rewards/oah33/DDQN2/20220423-135521/episode_900.csv" )