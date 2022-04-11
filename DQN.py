# To access Tensorboard in VS Code:
#       CTRL+Shift+p : Python: Launch Tensorboard

import gym
# from env_mod.car_racing_mod import CarRacing
#from gym import wrappers

import numpy as np
import cv2
from keras import Model, Input
from keras.models import Sequential
from keras.layers import Concatenate
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import random
from scipy import stats
import tensorflow as tf
import datetime
import os
from tensorflow.keras import datasets, layers, models
# import pyvirtualdisplay

bool_do_not_quit = True  # Boolean to quit pyglet
scores = []  # Your gaming score
a = np.array( [0.0, 0.0, 0.0] )  # Actions
prev_err = 0 

# Setup TensorBoard model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Creates a virtual display for OpenAI gym
# pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

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

        input_layer = tf.keras.Input(shape=(49))
        flat = tf.keras.layers.Flatten()(input_layer)
        dense1 = tf.keras.layers.Dense(64,activation='relu')(flat)
        dense2 = tf.keras.layers.Dense(32,activation='relu')(dense1)
        output_layer = tf.keras.layers.Dense(self.number_of_actions,activation='relu')(dense2)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='MSE', optimizer='adam')

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
            self.model.fit(state,action_vals,epochs=1,verbose=0, callbacks=[tensorboard_callback])  # note TensorBoard callback!


def image_processing(state):
    observation = state[63:65, 24:73]
       #convert to hsv
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    #slice the green
    imask_green = mask_green>0
    green = np.zeros_like(observation, np.uint8)
    green[imask_green] = observation[imask_green]
    gray = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny[0]

def train_agent(episodes):
    env = gym.make('CarRacing-v0').env
    # env =  CarRacing()
    #env = wrappers.Monitor(env, '/homes/oah33/Reinforcement-Learning-G69', video_callable=False ,force=True)

    for episodeNum in range(episodes):
        print("episode:",episodeNum)
        env.reset()  
        done = False
        action = (0,0,0)
        state_access = False
        step = 0
        reward_cum =0
        while not done and reward_cum  > -1:  
            # print("episode:",episodeNum,"step:",step)
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
    env.close()
    
agent = dnq_agent(epsilon=0.2,n=20)
print("here",agent.possible_actions[0])
print(agent.number_of_actions)
train_agent(100)
