from collections import deque
import random
import numpy as np


class ReplayBuffer():
    def __init__(self, buffer_size=128, random_seed=123):
        self._max_buffer_size = buffer_size
        self._count = 0
        self._buffer = deque()
        random.seed(random_seed)

    def add_sample(self, current_state, action, reward, terminal, next_state):
        experience = (current_state, action, reward, terminal, next_state)
        if self._count < self._max_buffer_size:
            self._buffer.append(experience)
            self._count += 1
        else:
            self._buffer.popleft()
            self._buffer.append(experience)

    def get_batch(self, batch_size):
        count = self._count if self._count < batch_size else batch_size
        batch = random.sample(self._buffer, count)

        state_batch      = np.array([el[0] for el in batch])
        action_batch     = np.array([el[1] for el in batch])
        reward_batch     = np.array([el[2] for el in batch])
        terminal_batch   = np.array([el[3] for el in batch])
        next_state_batch = np.array([el[4] for el in batch])

        return state_batch, action_batch, reward_batch, terminal_batch, next_state_batch

    def size(self):
        return len(self._buffer)

    def clear(self):
        self.deque.clear()
