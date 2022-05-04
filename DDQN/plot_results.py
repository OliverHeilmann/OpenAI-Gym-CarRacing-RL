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
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [2, 1, 2]})

    # plot random agent, human and PID on ax1 and ax3
    x = [0, 1900]
    dataDict = {    "Random" : [-24.28] * len(x),
                    "Human" : [770.75] * len(x),
                    "PID" : [680.98] * len(x) }

    labels = []
    for key, val in dataDict.items():
        ax1.plot( x, val, '--', linewidth=1 )
        ax3.plot( x, val, '--', linewidth=1 )
        labels.append( key )

    for label, tpath, mpath in filepaths:
        # create a meaningful legend
        labels.append( label )

        # read data and plot as SMA
        data12 = pd.read_csv(tpath, names=["Reward", "Epsilon", "Run Time"])  
        reward = data12["Reward"].to_numpy()
        epsilon = data12["Epsilon"].to_numpy()
        episodes = np.array( range(len(reward)))
        sma_reward, ticks = simple_moving_avg( reward )

        # ax1.plot(episodes, reward, 'k-', linewidth=1)
        ax1.plot(episodes, sma_reward, '-', linewidth=1)
        ax1.set(title=f'{ticks} Period Simple Moving Average of Training Reward Against Episode', ylabel='Reward')

        ax2.plot(episodes, epsilon, '-', linewidth=1)
        ax2.set(title=f'Epsilon Against Episode', ylabel='Epsilon')

        # now get data for plot 3s
        if [tpath, mpath] != filepaths[-1]:
            data3 = pd.read_csv(mpath, names=["Episode", "Avg Reward", "Epsilon", "Avg Time", "Max", "Min", "Std Dev"])  
            ax3.plot(data3["Episode"], data3["Avg Reward"], '-', linewidth=1)
        else:
            data3 = pd.read_csv(mpath, names=["Avg Reward", "Epsilon", "Avg Time", "None", "Max", "Min", "Std Dev"])
            ax3.plot(list(range(len(data3["Avg Reward"]))), data3["Avg Reward"], '-', linewidth=1)
    
    ax3.set(title=f'Testing Reward Against Episode', xlabel='Training Episode', ylabel='Avg Reward (50 Runs)')
    ax3.legend( labels, bbox_to_anchor=(0,-1,1,1), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    fig.set_size_inches(7,8)
    fig.tight_layout()  # add padding between figs
    # plt.show()
    plt.savefig('imgs/results.png')


if __name__ == '__main__':

    labels =    ["DQN",
                "DDQN Termination Type 1",
                "DDQN Termination Type 2",
                "Prioritised DDQN",
                # "Prioritised DDQN Big R.B.",
                "DDPG",
                "DDPG Reward Clipping"
                ]

    # MUST BE SAME LENGTH!
    training_rewards = [   "DDQN/rewards/oah33/DQN2/20220422-164216/episode_1200.csv",
                            # "DDQN/rewards/oah33/DDQN1/20220422-190009/episode_300.csv",
                            "DDQN/rewards/oah33/DDQN2/20220423-122311/episode_1900.csv",
                            "DDQN/rewards/oah33/DDQN2/20220423-170444/episode_1900.csv",
                            "DDQN/rewards/oah33/DDQN3_NN/20220424-140943/episode_1900.csv",
                            # "DDQN/rewards/oah33/DDQN3_NN_BigBuffer/20220427-115058/episode_500.csv",
                            # "DDQN/rewards/oah33/DDQN3_NN_BigBuffer/20220424-140943/episode_1900.csv",
                            "DDPG/rewards/20220503-211440/episode_1400.csv",
                            "DDPG/rewards/20220504-084112/episode_1400.csv",
                        ]

    model_rewards = [   "DDQN/episode_test_runs/oah33/20220425-202418/DQN2/episode_run_rewards.csv",
                        # "DDQN/episode_test_runs/oah33/20220425-170036/DDQN1/episode_run_rewards.csv",
                        "DDQN/episode_test_runs/oah33/20220425-202418/DDQN2_T1/episode_run_rewards.csv",
                        "DDQN/episode_test_runs/oah33/20220425-202418/DDQN2_T2/episode_run_rewards.csv",
                        "DDQN/episode_test_runs/oah33/20220425-202418/DDQN3_NN/episode_run_rewards.csv",
                        # "DDQN/episode_test_runs/oah33/20220425-202418/DDQN3_NN/episode_run_rewards.csv",
                        # "DDQN/episode_test_runs/oah33/20220504-104929/DDQN3_BigRB/episode_run_rewards.csv",
                        "DDPG/test rewards/20220504-074748/episode_49.csv",
                        "DDPG/test rewards/20220504-173711/run2.csv",
                    ]
    filepaths = [ [labels[i], training_rewards[i], model_rewards[i]] for i in range(len(training_rewards)) ]
    plotResults( filepaths = filepaths )