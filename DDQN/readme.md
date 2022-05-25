# Section Description
This section contains all code used to develop the final version of the DDQN algorithm presented in the `Before/ After Training` video and `Group Report`. Additionally, the relevant experiment data has been provided as well; this includes [output models](model/oah33/DDQN3_NN/20220424-140943/episode_900.h5), [training rewards vs episode](rewards/oah33/DDQN3_NN/20220424-140943/episode_900.csv) and [average testing rewards vs episode](episode_test_runs/oah33/20220425-202418/DDQN3_NN/episode_run_rewards.csv).

# Code Usage

## Reinforcement Learning Models
All reinforcement learning scripts in this directory ([DQN1](DDQN1.py), [DQN2](DDQN2.py), [DDQN1](DDQN1.py), [DDQN2](DDQN2.py), [DDQN3](DDQN3.py)) have a clearly indicated `Configuration` section at the top of each file. As the [HEX Cloud](https://hex.cs.bath.ac.uk/) was used for some training, some configuration lines have been included. Additionally, users may add their `username` and `model type`; this is used to define a directory for the model weights and reward vs episode data to be saved to.

```python
############################## CONFIGURATION ##################################
# Prevent tensorflow from allocating the all of GPU memory
# From: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
GPUs = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUs:
    tf.config.experimental.set_memory_growth( gpu, True )   # set memory growth option

# Creates a virtual display for OpenAI gym ( to support running from headless servers)
pyvirtualdisplay.Display( visible=0, size=(720, 480) ).start()

# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "STUDENT_USERNAME"
MODEL_TYPE              = "MODEL_NAME"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR               = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup Reward Dir
REWARD_DIR              = f"rewards/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"
```

Additionally, users may tune the model hyper-parameters by simply modifying the variables shown in the `Configuration` section. Should one wish to render the environment during training, `RENDER` should be set to `True` for instance. Should a user wish to test a model's optimal policy, they may simply set the `PRETRAINED_PATH` variable as the path to the saved weights – note that users should also set `RENDER` and `TEST` to `True` (if `TEST == False` then the default mode is *training*).

``` python
# Training params
RENDER                  = True
PLOT_RESULTS            = False     # plotting reward and epsilon vs epsiode (graphically) NOTE: THIS WILL PAUSE TRAINING AT PLOT EPISODE!
EPISODES                = 2000      # training episodes
SAVE_TRAINING_FREQUENCY = 100       # save model every n episodes
SKIP_FRAMES             = 2         # skip n frames between batches
TARGET_UPDATE_STEPS     = 5         # update target action value network every n EPISODES
MAX_PENALTY             = -30       # min score before env reset
BATCH_SIZE              = 20        # number for batch fitting
CONSECUTIVE_NEG_REWARD  = 25        # number of consecutive negative rewards before terminating episode
STEPS_ON_GRASS          = 20        # How many steps can car be on grass for (steps == states)
REPLAY_BUFFER_MAX_SIZE  = 150000    # threshold memory limit for replay buffer (old version was 10000)

# Testing params
PRETRAINED_PATH         = "model/oah33/DDQN3_NN_BigBuffer/20220427-115058/episode_400.h5"
TEST                    = True      # true = testing, false = training
```

That is it! Provided the user has the appropriate libraries installed, the aforementioned scripts should run!

## Testing Model
The [test_model.py](test_model.py) script is used to evaluate the *testing performance* of multiple models; in other words, calculating the average reward earned by an agent using their optimal policy. This script also supports testing multiple differing algorithms as well as different episode weights.

The following lines of code can be modified to allow users to test differing agents. In this example, the the DQN and DDQN algorithm types are all being tested sequentially, each of which having multiple model weights i.e. `DQN2` has 12 model weights in the `20220422-164216` folder, each of which will be tested over 50 runs (and so on).
```python
agents_functs_folders = [   ["DQN2", DQN_Agent2, DQN2_convert_greyscale, "DDQN/model/oah33/DQN2/20220422-164216"],
                            # ["DDQN1", DDQN_Agent1, DDQN1_convert_greyscale, "DDQN/model/oah33/DDQN1/20220422-190009"],
                            ["DDQN2_T1", DDQN_Agent2, DDQN2_convert_greyscale, "DDQN/model/oah33/DDQN2/20220423-122311"],
                            ["DDQN2_T2", DDQN_Agent2, DDQN2_convert_greyscale, "DDQN/model/oah33/DDQN2/20220423-170444"],
                            ["DDQN3_NN", DDQN_Agent3, DDQN3_convert_greyscale, "DDQN/model/oah33/DDQN3_NN/20220424-140943"],
                        ]
```

## Plotting Results
The [plot_results](plot_results.py) script allows users to plot the data gathered from training and testing their agents. Similar to above, there is a small degree of configuration to be done before plotting results. The example below shows how the training rewards array takes four string entries (directories) to the relevant data.

```python
# MUST BE SAME LENGTH!
training_rewards = [   "DDQN/rewards/oah33/DQN2/20220422-164216/episode_1200.csv",
                        # "DDQN/rewards/oah33/DDQN1/20220422-190009/episode_300.csv",
                        "DDQN/rewards/oah33/DDQN2/20220423-122311/episode_1900.csv",
                        "DDQN/rewards/oah33/DDQN2/20220423-170444/episode_1900.csv",
                        "DDQN/rewards/oah33/DDQN3_NN/20220424-140943/episode_1900.csv",
                        # "DDQN/rewards/oah33/DDQN3_NN_BigBuffer/20220427-115058/episode_500.csv"
                    ]

model_rewards = [   "DDQN/episode_test_runs/oah33/20220425-202418/DQN2/episode_run_rewards.csv",
                    # "DDQN/episode_test_runs/oah33/20220425-170036/DDQN1/episode_run_rewards.csv",
                    "DDQN/episode_test_runs/oah33/20220425-202418/DDQN2_T1/episode_run_rewards.csv",
                    "DDQN/episode_test_runs/oah33/20220425-202418/DDQN2_T2/episode_run_rewards.csv",
                    "DDQN/episode_test_runs/oah33/20220425-202418/DDQN3_NN/episode_run_rewards.csv",
                    # "DDQN/episode_test_runs/oah33/20220425-202418/DDQN3_NN/episode_run_rewards.csv"
                ]
```

The output may look something like the below:

![DQN, DDQN](/imgs/results.png "Discrete Action Space RL Approaches")

