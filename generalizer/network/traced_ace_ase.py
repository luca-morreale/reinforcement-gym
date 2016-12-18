
import numpy as np
from collections import deque

from adaptive_ace_ase import ACEASE
from updater.trace import Trace

class TracedACEASE(ACEASE):
    def __init__(self, session, state_dim, gamma=0.8, learning_rate=0.1, _lambda=0.8):
        ACEASE.__init__(self, session, state_dim, gamma, learning_rate)
        self._lambda = _lambda
        self.trace = [deque(), np.zeros(shape=(1))]
        self.trace[0].append(0)

    def cleanup_trace(self):

        while self.trace[1][0] <= 0.0009:
            self.trace[0].popleft()
            self.trace[1] = np.delete(self.trace[1], 0)

    def _update_trace(self, experience):
        self.trace[0].append(experience)
        self.trace[1] *= self._lambda
        self.trace[1] = np.append(self.trace[1], np.array([1]), axis=0)

    def train(self, state, action, reward, done, next_state):
        experience = (self.critic.reshape_state(state), np.array([reward]), done) # tuple state - reward
        self._update_trace(experience)
        self.cleanup_trace()

        rewards, states, terminals, trace = self.extract_from_trace()

        deltas = []
        for k in xrange(len(trace)):
            if terminals[k]:
                deltas.append(reward[k][0])
            else:
                deltas.append(self.critic.get_internal_signal(states[k], rewards[k])[0])
        actor_target, critic_target, states = self.prepare_batches(deltas, rewards, states, trace)

        # batch update
        self._train_networks(states, actor_target_out=actor_target, critic_target_out=critic_target)

    def extract_from_trace(self):
        tuples = self.trace[0]
        states = [elem[0] for elem in tuples]
        rewards = [elem[1] for elem in tuples]
        terminals = [elem[2] for elem in tuples]
        return rewards, states, terminals, self.trace[1]

    def prepare_batches(self, deltas, rewards, states, trace):
        actor_target = np.transpose(np.multiply(np.transpose(deltas), trace))
        critic_target = np.transpose(np.multiply(np.transpose(rewards), trace))
        states = self.transform_into_batch(states, len(trace))
        return actor_target, critic_target, states

    def transform_into_batch(self, state_batch, length):
        state_shape = list(self._state_shape())
        state_shape[0] = length
        return np.reshape(state_batch, tuple(state_shape))