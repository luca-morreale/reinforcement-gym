from ace_net import ACE
from ase_net import ASE

import tensorflow as tf
import numpy as np

# Based on paper 'Neuronlike adaptive elements that can solve difficult learning control problems'
# of Barto, Sutton and Anderson

class ACEASE():
    def __init__(self, session, state_dim, gamma=0.8, learning_rate=0.1):
        self.critic = ACE(session, state_dim, gamma, learning_rate)
        self.actor = ASE(session, state_dim, learning_rate)
        self._state_dim = state_dim
        self._session = session
        self._session.run(tf.initialize_all_variables())

    def train(self, state, reward):
        delta = self.critic.get_internal_signal(np.reshape(state, self._state_shape()), reward)
        state = np.reshape(state, self._state_shape())
        self._train_networks(state, actor_target_out=delta, critic_target_out=reward)

    def _train_networks(self, state, actor_target_out, critic_target_out):
        self.actor.train(state, actor_target_out)
        self.critic.train(state, critic_target_out)

    def get_action(self, state):
        return self.actor.predict(np.reshape(state, self._state_shape()))

    def transform_into_batch(self, state_batch, length):
        state_shape = list(self._state_shape())
        state_shape[0] = length
        return np.reshape(state_batch, tuple(state_shape))