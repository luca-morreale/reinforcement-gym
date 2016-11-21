
from regression_network import RegressionNetwork

import tensorflow as tf

# Predict action

class ASE(RegressionNetwork):

    def __init__(self, session, state_dim, action_dim, learning_rate=0.1):
        RegressionNetwork.__init__(self, session, state_dim, learning_rate=learning_rate)
        self._action_dim = action_dim

    def train(self, inputs, delta):
        prediction = RegressionNetwork.predict(self, inputs)
        return RegressionNetwork.train(self, inputs, delta + prediction)
