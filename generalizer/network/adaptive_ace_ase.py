from ace_net import ACE
from ase_net import ASE

import tensorflow as tf
import numpy as np

# Based on paper 'Neuronlike adaptive elements that can solve difficult learning control problems'
# of Barto, Sutton and Anderson

class ACEASE():
    def __init__(self, session, state_dim, action_dim, gamma=0.8, learning_rate=0.1):
        self.critic = ACE(session, state_dim, gamma, learning_rate)
        self.actor = ASE(session, state_dim, action_dim, learning_rate)
        self._state_dim = state_dim
        self._session = session
        self._session.run(tf.initialize_all_variables())

    def update_nets(self, state, reward):
        for i in range(10):
            delta = self.critic.get_internal_signal(np.reshape(state, self._state_shape()), reward)
            self.actor.train(np.reshape(state, self._state_shape()), delta)
            self.critic.train(np.reshape(state, self._state_shape()), delta)

    def _state_shape(self):
        return tuple(reversed( (self._state_dim, ) + (1,) ))

    def get_action(self, state):
        return self.actor.predict(np.reshape(state, self._state_shape()))
