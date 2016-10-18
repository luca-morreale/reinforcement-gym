
from network import NetworkGeneralizer

import tensorflow as tf
import numpy as np

# Predict action

class ASE(NetworkGeneralizer):

    def __init__(self, session, state_dim, action_dim, learning_rate=0.1):
        NetworkGeneralizer.__init__(self, session, state_dim, learning_rate=learning_rate)
        self._action_dim = action_dim
        self._scaled_out = tf.mul(self.net_y, self._action_dim)


    def train(self, inputs, delta):
        prediction = NetworkGeneralizer.predict(self, inputs)
        return NetworkGeneralizer.train(self, inputs, delta + prediction)

    def predict(self, inputs):
        return self._session.run(self._scaled_out, feed_dict={
            self.net_x: inputs
        })