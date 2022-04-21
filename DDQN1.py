# DQN version 2 to improve on the performance of DQN1.py
# Changes:
# 1. ...
# 2. ...
# 3. ...

# Environment imports
import random
import numpy as np
import gym
import pyvirtualdisplay
import cv2
from scipy import stats

# Tensorflow training imports
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Eager execution to speed up training speeds (A LOT!)
tf.compat.v1.disable_eager_execution()

# Training monitoring imports
import datetime, os
from tqdm import tqdm
import time


############################## SERVER CONFIGURATION ##################################
# Prevent tensorflow from allocating the all of GPU memory
# From: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
GPUs = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUs:
    tf.config.experimental.set_memory_growth( gpu, True )   # set memory growth option

# Creates a virtual display for OpenAI gym ( to support running from headless servers)
pyvirtualdisplay.Display( visible=0, size=(720, 480) ).start()

# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "oah33"
MODEL_TYPE              = "DDQN1"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR               = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup Reward Dir
REWARD_DIR              = f"rewards/{TIMESTAMP}/"

# Training params
RENDER                  = True
EPISODES                = 2000      # training episodes
SAVE_TRAINING_FREQUENCY = 10        # save model every n episodes
SKIP_FRAMES             = 2         # skip n frames between batches
TARGET_UPDATE_STEPS     = 5         # update target action value network every n steps
MAX_PENALTY             = -1        # min score before env reset

# Testing params
PRETRAINED_PATH         = "model/oah33/DQN2/20220421-132428/episode_50.h5"
TEST                    = False      # true = testing, false = training


############################## MAIN CODE BODY ##################################
class DDQN_Agent:
    def __init__(   self, 
                    action_space    = [
                    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
                    (-1, 1,   0), (0, 1,   0), (1, 1,   0), #           (Steering, Gas, Break)
                    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range       -1~1     0~`1`   0~1
                    (-1, 0,   0), (0, 0,   0), (1, 0,   0)
                    ],
                    memory_size     = 10000,     # threshold memory limit for replay buffer
                    batch_size      = 10,       # number for batch fitting
                    gamma           = 0.95,     # discount rate
                    epsilon         = 1.0,      # exploration rate
                    epsilon_min     = 0.00025,  # used by Atari
                    epsilon_decay   = 0.9999,
                    learning_rate   = 0.001
                ):
        
        self.action_space    = action_space
        self.D               = deque( maxlen=memory_size )
        self.batch_size      = batch_size
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        
        # clone the action value network to make target action value network
        self.model           = self.build_model()
        self.target_model    = tf.keras.models.clone_model( self.model )

    def build_model( self ):
        """Sequential Neural Net with x2 Conv layers, x2 Dense layers using RELU and Huber Loss"""
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(81, 96, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        # model.add(Dense(54, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        return model

    def update_model( self ):
        """Update Target Action Value Network to be equal to Action Value Network"""
        self.target_model.set_weights( self.model.get_weights() )

    def store_transition( self, state, action, reward, new_state, done ):
        """Store transition in the replay memory (for replay buffer)."""
        self.D.append( (state, action, reward, new_state, done) )

    def choose_action( self, state ):
        """Take state input and use latest target model to make prediction on best next action; choose it!"""
        state = np.expand_dims(state, axis=0)
        actions = list( self.target_model.predict(state)[0] )
        actionIDX = actions.index( max(actions) )

        # epsilon chance to choose random action
        if stats.bernoulli( self.epsilon ).rvs():
            actionIDX = random.randint( 0, len(self.action_space)-1 )
        return self.action_space[ actionIDX ]

    def experience_replay( self ):
        """Use experience_replay with batch fitting and epsilon decay."""
        if len( self.D ) >= self.batch_size:
            # batch sample size
            minibatch = random.sample( self.D, self.batch_size )

            # experience replay
            train_state = []
            train_target = []
            for state, action, reward, next_state, done in minibatch:
                target = self.model.predict(np.expand_dims(state, axis=0))[0]
                if done:
                    target[ self.action_space.index(action) ] = reward
                else:
                    ############ Double Deep Q Learning Here! #############
                    # get index of action value network prediction for best action at next state
                    t = self.model.predict(np.expand_dims(next_state, axis=0))[0]
                    t_index = np.where(t == np.amax(t))[0][0]

                    # get target network prediction for next state, then use index calc'd above to
                    # update Q action value network
                    target_t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                    target[ self.action_space.index(action) ] = reward + self.gamma * target_t[ t_index ]
                
                train_state.append(state)
                train_target.append(target)

            # batch fitting
            self.model.fit( np.array(train_state), np.array(train_target), epochs=1, verbose=0 )
            
            # epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save( self, name, rewards ):
        """Save model and rewards list to appropriate dir, defined at start of code."""
        if not os.path.exists( MODEL_DIR ):
             os.makedirs( MODEL_DIR )
        self.model.save_weights( MODEL_DIR + name + ".h5" )

        if not os.path.exists( REWARD_DIR ):
             os.makedirs( REWARD_DIR )
        np.savetxt(f"{REWARD_DIR}" + name + ".csv", rewards, delimiter=",")

    def load( self, name ):
        """Load previously trained model weights."""
        self.model.load_weights( name )
        self.model.set_weights( self.model.get_weights() )

def convert_greyscale( state ):
    """Take input state and convert to greyscale. Check if road is visible in frame."""
    x, y, _ = state.shape
    state = state[ 0:int( 0.85*y ) , 0:x ]
    mask = cv2.inRange( state,  np.array([100, 100, 100]),  # dark_grey
                                np.array([150, 150, 150]))  # light_grey
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    # returns [ greyscale image, T/F of if road is visible ]
    return [ np.expand_dims( gray, axis=2 ), np.any(mask== 255) ]

def train_agent( agent : DDQN_Agent, env : gym.make, episodes : int ):
    """Train agent with experience replay, batch fitting and using a cropped greyscale input image."""
    episode_rewards = []
    for episode in tqdm( range(episodes) ):
        print( f"[INFO]: Starting Episode {episode}" )
        
        state_colour = env.reset() 
        state_grey, can_see_road = convert_greyscale( state_colour )

        sum_reward = 0
        step = 0
        done = False
        while not done and sum_reward > MAX_PENALTY and can_see_road:
            # update target action value network every N steps ( to equal action value network)
            if step % TARGET_UPDATE_STEPS == 0:
                agent.update_model()

            # choose action to take next
            action = agent.choose_action( state_grey )

            # take action and observe new state, reward and if terminal.
            new_state_colour, reward, done, _ = env.step( action )
            
            # convert to greyscale for NN
            new_state_grey, can_see_road = convert_greyscale( new_state_colour )

            # render if user has specified
            if RENDER: env.render()

            # if steps % SKIP_FRAMES == 0 then add the data to training
            if step % SKIP_FRAMES == 0:
                # store transition states for experience replay
                agent.store_transition( state_grey, action, reward, new_state_grey, done )

            # do experience replay
            agent.experience_replay()

            # update params for next loop
            state_grey = new_state_grey
            sum_reward += reward
            step += 1

        # Store episode reward
        episode_rewards.append( [sum_reward, agent.epsilon] )

        if episode % SAVE_TRAINING_FREQUENCY == 0:
            agent.save(f"episode_{episode}", rewards = episode_rewards)
    env.close()


def test_agent( agent : DDQN_Agent, env : gym.make, model : str ):
    """Test a pretrained model and print out run rewards and total time taken. Quit with ctrl+c."""
    # Load agent model
    agent.load( model )
    run_rewards = []
    while True:
        state_colour = env.reset() 
        state_grey, _ = convert_greyscale( state_colour )

        sum_reward = 0.0
        t1 = time.time()  # Trial timer
        while sum_reward > MAX_PENALTY:

            # choose action to take next
            action = agent.choose_action( state_grey )
            
            # take action and observe new state, reward and if terminal
            new_state_colour, reward, done, _ = env.step( action )

            # render if user has specified
            if RENDER: env.render()

            # convert to greyscale for NN
            new_state_grey, _ = convert_greyscale( new_state_colour )

            # update state
            state_grey = new_state_grey

            sum_reward += reward

        t1 = time.time()-t1
        run_rewards.append( sum_reward )
        run_rewards.append( sum_reward )
        
        print("[INFO]: Run Reward: ", sum_reward, " | Time:", "%0.2fs."%t1 )


if __name__ == "__main__":
    env = gym.make('CarRacing-v0').env

    if not TEST:
        # Train Agent
        agent = DDQN_Agent()
        train_agent( agent, env, episodes = EPISODES )
    
    else:
        # Test Agent
        agent = DDQN_Agent()
        test_agent( agent, env, model = PRETRAINED_PATH )