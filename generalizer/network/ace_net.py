
import tensorflow as tf

from regression_network import RegressionNetwork

# Predict the q value

class ACE(RegressionNetwork):
    def __init__(self, session, state_dim, gamma=0.8, learning_rate=0.1):
        RegressionNetwork.__init__(self, session, state_dim, learning_rate=learning_rate)
        self.old_net_x, self.old_net_y, self.old_weights = self._create_clone_net()
        self.initialized = False
        self._gamma = gamma

    def _create_clone_net(self):
        old_net_x, old_net_y, old_weights = self._create_net()
        old_net_y, w = self._add_linear_out(old_net_y)
        old_weights.append(w)
        return  old_net_x, old_net_y, old_weights

    def _copy_net(self):
        operations = []
        weight_size = len(self.weights)
        for i in range(weight_size):
            operations.append(self.old_weights[i].assign(self.weights[i]))

        self._session.run(operations)
        self.initialized = True

    def get_internal_signal(self, inputs, reward):
        signal = 0
        if self.initialized == False:
            signal = reward
        else:
            prediction = self.predict(inputs)
            old_prediction = self.old_prediction(inputs)
            signal = reward + self._gamma * prediction - old_prediction

        return signal

    def train(self, inputs, delta):
        prediction = self.predict(inputs)
        self._copy_net()
        return RegressionNetwork.train(self, inputs, prediction + delta)

    def old_prediction(self, inputs):
        return self._session.run(self.old_net_y, feed_dict={
            self.old_net_x: inputs
        })
