# To access Tensorboard in VS Code:
#       CTRL+Shift+p : Python: Launch Tensorboard

import gym
# from env_mod.car_racing_mod import CarRacing
import numpy as np
import cv2
from keras import Model, Input
from keras.models import Sequential
from keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import random
from scipy import stats
import tensorflow as tf
import datetime
import os
from tensorflow.keras import datasets, layers, models
import pyvirtualdisplay


############################## SERVER CONFIGURATION ##################################

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
SAVE_TRAINING_FREQUENCY = 1
model_dir = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup TensorBoard model
# log_dir = f"logs/fit/{TIMESTAMP}"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Eager execution
tf.compat.v1.disable_eager_execution()


############################## MAIN CODE BODY ##################################
bool_do_not_quit = True  # Boolean to quit pyglet
scores = []  # Your gaming score
a = np.array( [0.0, 0.0, 0.0] )  # Actions
prev_err = 0 
class dnq_agent:
    def __init__(self,epsilon,n):
        self.D = []
        self.epsilon = epsilon
        self.n = n

        self.possible_actions = []

        for steer in [0,0.2,0.4,0.6,0.8,1]:
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


    def make_move(self,state):
        state = np.expand_dims(state, axis=0)
        actions = list(self.model.predict(state)[0])
        actionIDX = actions.index(max(actions))
        if stats.bernoulli(self.epsilon).rvs():
            actionIDX = random.randint(0, self.number_of_actions-1)
        return actionIDX

    def make_observation(self,state,action,reward,new_state):
        self.D.append((state,action,reward,new_state))
        if len(self.D) > 10000:
            self.D.pop(0)

    def learn_from_D(self):
        for _ in range(self.n):
            (state,action,reward,new_state) = random.choice(self.D)
            
            new_state = np.expand_dims(new_state, axis=0)
            state = np.expand_dims(state, axis=0)

            max_val = max(list(self.model.predict(new_state)[0]))
            y = reward + max_val

            action_vals = self.model.predict(state)[0]
            action_vals[action] = y
            action_vals = np.expand_dims(action_vals , axis=0)
            self.model.fit(state,action_vals,epochs=1,verbose=0) #, callbacks=[tensorboard_callback])  # note TensorBoard callback!

    def save(self, name):
        """Save model to appropriate dir, defined at start of code."""
        if not os.path.exists(model_dir):
             os.makedirs(model_dir)
        self.model.save_weights(model_dir + name)


def image_processing(state):
    x, y, _ = state.shape
    h = int( 0.9*y ); w = x
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[0:h, 0:w]
    crop_gray = np.expand_dims(crop_gray, axis=2)
    return crop_gray

def train_agent(episodes):
    env = gym.make('CarRacing-v0').env
    # env =  CarRacing()

    for episodeNum in range(episodes):
        print("[INFO]: Episode:", episodeNum )
        env.reset()  
        done = False
        action = (0,0,0)
        state_access = False
        step = 0
        reward_cum =0
        while not done and reward_cum  > -1:  
            step+=1
            state, reward, done, info = env.step(action)
            reward_cum += reward
            procesed_image = image_processing(state) 

            if state_access:
                agent.make_observation(state=prev_state,action=action_idx,reward=reward,new_state=procesed_image)
                agent.learn_from_D()

            prev_state = procesed_image
            state_access =True
            action_idx = agent.make_move(procesed_image)
            action = agent.possible_actions[action_idx]   

        if episodeNum % SAVE_TRAINING_FREQUENCY == 0:
            agent.save(f"episode_{episodeNum}.h5")

    env.close()
    
agent = dnq_agent(epsilon=0.2,n=20)
train_agent(100)
