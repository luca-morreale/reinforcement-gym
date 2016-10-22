from adaptive_ace_ase import ACEASE
from replay_buffer import ReplayBuffer

import numpy as np


class BufferedACEASE(ACEASE):

    def __init__(self, session, state_dim, action_dim, gamma=0.8, learning_rate=0.1, bufferSize=1024):
        ACEASE.__init__(self, session, state_dim, action_dim, gamma, learning_rate)
        self.buffer = ReplayBuffer(bufferSize)
        self.MINI_BATCH_SIZE = 128

    def train(self, state, action, reward, terminal, next_state):

        self.buffer.add_sample(np.reshape(state, self._state_shape()), action, reward, terminal,
                                    np.reshape(next_state, self._state_shape()))

        if self.buffer.size() > self.MINI_BATCH_SIZE:
            state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = self.buffer.get_batch(self.MINI_BATCH_SIZE)

            deltas = []
            for k in xrange(self.MINI_BATCH_SIZE):
                    if terminal_batch[k]:
                        deltas.append(reward_batch[k])
                    else:
                        deltas.append(self.critic.get_internal_signal(state_batch[k], reward_batch[k]))

            for k in xrange(self.MINI_BATCH_SIZE):
                self.actor.train(state_batch[k], deltas[k])
                self.critic.train(state_batch[k], deltas[k])
