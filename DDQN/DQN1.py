# To access Tensorboard in VS Code:
#       CTRL+Shift+p : Python: Launch Tensorboard

import gym
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import random
from scipy import stats
import tensorflow as tf
import pyvirtualdisplay
import datetime, os
from tqdm import tqdm


############################## CONFIGURATION ##################################

# Prevent tensorflow from allocating the all of GPU memory
# From: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)   # set memory growth option
    # tf.config.set_logical_device_configuration( gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=3000)] )    # set memory limit to 3 GB

# Creates a virtual display for OpenAI gym
pyvirtualdisplay.Display(visible=0, size=(720, 480)).start()

# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "oah33"
MODEL_TYPE              = "DQN"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVE_TRAINING_FREQUENCY = 10
model_dir = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup TensorBoard model
log_dir = f"logs/fit/{TIMESTAMP}"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Setup Reward Dir
reward_dir = f"rewards/{TIMESTAMP}/"

# Eager execution
tf.compat.v1.disable_eager_execution()


############################## MAIN CODE BODY ##################################
bool_do_not_quit = True  # Boolean to quit pyglet
scores = []  # Your gaming score
a = np.array( [0.0, 0.0, 0.0] )  # Actions
prev_err = 0
EPISODES = 1000

class dnq_agent:
    def __init__(self,epsilon,n,gamma):
        self.D = []
        self.epsilon = epsilon
        self.n = n
        self.gamma = gamma

        self.possible_actions = []

        for steer in [0,0.5,1]:
            for accel in [0,0.5,1]:
                self.possible_actions.append((steer,accel,0))
                self.possible_actions.append((-steer,accel,0))
        
        self.number_of_actions = len(self.possible_actions)

        # create DNN network with x2 conv layers
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(86, 96, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(216, activation='relu'))
        self.model.add(Dense(self.number_of_actions, activation=None))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, epsilon=1e-7))

        # create action model clone
        self.action_model = tf.keras.models.clone_model(self.model)

    def make_move(self,state):
        state = np.expand_dims(state, axis=0)
        actions = list(self.action_model.predict(state)[0])
        actionIDX = actions.index(max(actions))
        if stats.bernoulli(self.epsilon).rvs():
            actionIDX = random.randint(0, self.number_of_actions-1)

        return actionIDX

    def make_best_move(self, state):
        """Return best possible action"""
        state = np.expand_dims(state, axis=0)
        actions = list(self.action_model.predict(state)[0])
        return actions.index(max(actions))

    def make_observation(self,state,action,reward,new_state):
        self.D.append((state,action,reward,new_state))
        if len(self.D) > 10000:
            self.D.pop(0)
    
    def update_model(self):
        self.action_model = tf.keras.models.clone_model(self.model)

    def learn_from_D(self):
        X=[]
        Y=[]
        for _ in range(self.n):
            (state,action,reward,new_state) = random.choice(self.D) 
            X.append(state)
            max_val = max(list(self.model.predict(np.expand_dims(new_state, axis=0))[0]))
            y = reward + self.gamma*max_val
            action_vals = self.model.predict(np.expand_dims(state, axis=0))[0]
            action_vals[action] = y 
            Y.append(action_vals)
        X = np.array(X)
        Y = np.array(Y)  
        Y = np.clip(Y, a_min = -1, a_max = 1)
        self.model.fit(X,Y,epochs=1,verbose=0) #, callbacks=[tensorboard_callback])  # note TensorBoard callback!
        
    def save(self, name, reward):
        """Save model and rewards list to appropriate dir, defined at start of code."""
        if not os.path.exists(model_dir):
             os.makedirs(model_dir)
        self.model.save_weights(model_dir + name + ".h5")

        if not os.path.exists(reward_dir):
             os.makedirs(reward_dir)
        np.savetxt(f"{reward_dir}" + name + ".csv", reward, delimiter=",")

    def load(self, name):
        """Load previously trained model weights."""
        self.model.load_weights(name)
        self.model.set_weights( self.model.get_weights() )



def image_processing(state):
    x, y, _ = state.shape
    h = int( 0.9*y ); w = x
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[0:h, 0:w]
    crop_gray = np.expand_dims(crop_gray, axis=2)
    return crop_gray


def train_agent(episodes, start_from = None):
    # load model and continue training from episode X
    episode = 0
    if start_from:
        agent.load( start_from )
        model_dir = "./" + start_from.split("episode_")[0]  # overwrite model directory
        episode = int(start_from.split("episode_")[1].split(".")[0])    # get current episode number

    env = gym.make('CarRacing-v0').env
    total_reward = []

    for episodeNum in tqdm(range(episode+1, episodes)):
        rewards = []
        print("[INFO]: Starting Episode:", episodeNum )
        env.reset()  
        done = False
        action = (0,0,0)
        state_access = False
        step = 0
        reward_terminate = False
        reward_cum =0

        while not done and reward_cum  > -1:
            if step == 100:
                agent.update_model()
                step = 0
            step+=1

            # make action
            state, reward, done, info = env.step(action)
            reward_cum += reward

            env.render()

            # get cropped and grey image
            procesed_image = image_processing(state) 

            if state_access:
                agent.make_observation(state=prev_state,action=action_idx,reward=reward,new_state=procesed_image)
                agent.learn_from_D()
            prev_state = procesed_image
            state_access =True
            action_idx = agent.make_move(procesed_image)
            action = agent.possible_actions[action_idx]

        # Store episode reward
        total_reward.append( reward_cum )

        if episodeNum % SAVE_TRAINING_FREQUENCY == 0:
            agent.save(f"episode_{episodeNum}", reward = total_reward)
    env.close()

if __name__ == "__main__":
    agent = dnq_agent(epsilon=0.2,n=100,gamma=0.5)
    train_agent(EPISODES)#, start_from = "model/oah33/DQN/20220420-183322/episode_40.h5")