import numpy as np
from utils import *


class ReplayBuffer:
    def __init__(self, mem_size=50000):
        self.mem_size = mem_size

        # Memory will be initialized when first time used
        self.state_db = None
        self.action_db = None
        self.reward_db = None
        self.new_state_db = None

        self.writes_num = 0

    def init_memory(self, state_shape, action_shape):
        state_shape = prepend_tuple(self.mem_size, state_shape)
        action_shape = prepend_tuple(self.mem_size, action_shape)

        self.state_db = np.zeros(state_shape, np.float32)
        self.action_db = np.zeros(action_shape, np.float32)
        self.reward_db = np.zeros((self.mem_size, 1), np.float32)
        self.new_state_db = np.zeros(state_shape, np.float32)

    def save_move(self, state, action, reward, new_state):
        if self.state_db is None:
            self.init_memory(state.shape, action.shape)

        # Write indexes
        memory_index = self.writes_num % self.mem_size
        next_index = (self.writes_num + 1) % self.mem_size

        # Save next state to the same array with a next index
        self.state_db[memory_index] = state
        self.action_db[memory_index] = action
        self.reward_db[memory_index] = reward
        self.new_state_db[memory_index] = new_state

        self.writes_num += 1

    def sample_buffer(self, batch_size=64):
        indexes_range = min(self.mem_size, self.writes_num)
        sampled_indexes = np.random.choice(indexes_range, batch_size)

        return (self.state_db[sampled_indexes],
                self.action_db[sampled_indexes],
                self.reward_db[sampled_indexes],
                self.new_state_db[sampled_indexes])