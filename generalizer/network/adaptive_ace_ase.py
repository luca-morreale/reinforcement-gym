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
        self._session.run(tf.initialize_all_variables())

    def train(self, state, reward):
        state = self.critic.reshape_state(state)
        delta = self.critic.get_internal_signal(state, reward)
        self._train_networks(state, actor_target_out=delta, critic_target_out=reward)

    def _train_networks(self, state, actor_target_out, critic_target_out):
        self.actor.train(state, actor_target_out)
        self.critic.train(state, critic_target_out)

    def get_action(self, state):
        return self.actor.predict(self.actor.reshape_state(state))

    def reshape_state_batch(self, state_batch, length):
        return self.reshape_into_batch(state_batch, length, self.critic.state_shape())

    def reshape_into_batch(self, batch, length, shape):
        list_shape = list(shape)
        list_shape[0] = length
        return np.reshape(batch, tuple(list_shape))

