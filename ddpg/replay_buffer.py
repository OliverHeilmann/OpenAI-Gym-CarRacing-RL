import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, number_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, number_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.flag_memory = np.zeros(self.mem_size, dtype=np.bool)

    def save_move(self, state, action, reward, new_state, done):
        idx = self.mem_counter % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.flag_memory[idx] = done
        self.mem_counter += 1

    def smaple_buffer(self, batch_size):
        # how much of memory is filled up (ignoring 0s)
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        flags = self.flag_memory[batch]
        actions = self.action_memory[batch]

        return states, new_states, rewards, flags, actions



