import numpy as np
from utils import *

class ReplayBuffer:
    def __init__(self, mem_size=50000):
        self.mem_size = mem_size

        # Memory will be initialized when first time used
        self.states = None
        self.actions = None
        self.rewards = None
        self.new_states = None

        self.transitions_num = 0

    def init_memory(self, state_shape, action_shape):
        state_shape = prepend_tuple(self.mem_size, state_shape)
        action_shape = prepend_tuple(self.mem_size, action_shape)

        self.states = np.zeros(state_shape, np.float32)
        self.actions = np.zeros(action_shape, np.float32)
        self.rewards = np.zeros((self.mem_size, 1), np.float32)
        self.new_states = np.zeros(state_shape, np.float32)

    def save_move(self, state, action, reward, new_state):
        if self.states is None:
            self.init_memory(state.shape, action.shape)

        # Write indexes
        memory_index = self.transitions_num % self.mem_size
        next_index = (self.transitions_num + 1) % self.mem_size

        # Save next state to the same array with a next index
        self.states[memory_index] = state
        self.actions[memory_index] = action
        self.rewards[memory_index] = reward
        self.new_states[memory_index] = new_state

        self.transitions_num += 1

    def sample_buffer(self, batch_size=64):
        indexes_range = min(self.mem_size, self.transitions_num)
        sampled_indexes = np.random.choice(indexes_range, batch_size)

        return (self.states[sampled_indexes],
                self.actions[sampled_indexes],
                self.rewards[sampled_indexes],
                self.new_states[sampled_indexes])
