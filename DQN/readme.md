# Section Description

In this section documenters the implementation of the DQN algorithm. To run the script run `python DQN.py` however some paramaters should be configured beforehand. Se the setup section for more details. The output model as well as the rewards should be placed in there respective `model` folder and `rewards` folder. 

## Setup

Some core parameters need to be configured before the program can be run :

`RENDER = True` : Determines if the environment renders or not.

`PRETRAINED_PATH = "MODEL_PATH/*.h5"` : Provide the path to the the trained model.

`TEST = False` : Set `TEST` to `True` to run the trained model. set to `False` to train the model.  

To speed up training (and to also allow the program to run without a GUI) the line :
`pyvirtualdisplay.Display( visible=0, size=(720, 480) ).start()`

has been added to the script. If you would like to see the environment whist it is training or the model during testing please comment out this line.

Further tunning can be done by configuring the following parameters : 

``` python
USERNAME                = "jo642"
MODEL_TYPE              = "DNQ2_10K"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR               = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup Reward Dir
REWARD_DIR              = f"rewards/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Training params
PLOT_RESULTS            = False     # plotting reward and epsilon vs epsiode (graphically) NOTE: THIS WILL PAUSE TRAINING AT PLOT EPISODE!
EPISODES                = 1000       # training episodes
SAVE_TRAINING_FREQUENCY = 50        # save model every n episodes
SKIP_FRAMES             = 2         # skip n frames between batches
TARGET_UPDATE_STEPS     = 5         # update target action value network every n EPISODES
MAX_PENALTY             = -5        # min score before env reset
BATCH_SIZE              = 65        # number for batch fitting
CONSECUTIVE_NEG_REWARD  = 30        # number of consecutive negative rewards before terminating episode
```
## Plotting results
For details seen the DDQN readme.
