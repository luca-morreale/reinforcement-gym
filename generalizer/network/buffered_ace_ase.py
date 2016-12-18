from adaptive_ace_ase import ACEASE
from replay_buffer import ReplayBuffer

import numpy as np


class BufferedACEASE(ACEASE):

    def __init__(self, session, state_dim, gamma=0.8, learning_rate=0.1, bufferSize=1024):
        ACEASE.__init__(self, session, state_dim, gamma, learning_rate)
        self.buffer = ReplayBuffer(bufferSize)
        self.MINI_BATCH_SIZE = 128

    def train(self, state, action, reward, terminal, next_state):

        self.buffer.add_sample(self.critic.reshape_state(state), action, np.array([reward]), terminal,
                                    self.critic.reshape_state(next_state))

        if self.buffer.size() > self.MINI_BATCH_SIZE:
            state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = self.buffer.get_batch(self.MINI_BATCH_SIZE)

            deltas = []
            for k in xrange(self.MINI_BATCH_SIZE):
                    if terminal_batch[k]:
                        deltas.append(reward_batch[k][0])
                    else:
                        deltas.append(self.critic.get_internal_signal(state_batch[k], reward_batch[k])[0])

            state_batch = self.transform_into_batch(state_batch, self.MINI_BATCH_SIZE)

            # batch update
            self._train_networks(state_batch, actor_target_out=np.array(deltas), critic_target_out=np.array(reward_batch))
