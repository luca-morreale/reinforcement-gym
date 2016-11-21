
from regression_network import RegressionNetwork

# Predict the q value

class ACE(RegressionNetwork):
    def __init__(self, session, state_dim, gamma=0.8, learning_rate=0.1):
        RegressionNetwork.__init__(self, session, state_dim, learning_rate=learning_rate)
        self._gamma = gamma
        self._copy_current_net()

    def _copy_current_net(self):
        self.prev_net_x = self.net_x
        self.prev_net_y = self.net_y

    def get_internal_signal(self, inputs, reward):
        prediction = self.predict(inputs)
        old_prediction = self.old_prediction(inputs)

        return reward + self._gamma * prediction - old_prediction

    def train(self, inputs, delta):
        prediction = self.predict(inputs)
        self._copy_current_net()
        return RegressionNetwork.train(self, inputs, prediction + delta)

    def old_prediction(self, inputs):
        return self._session.run(self.prev_net_y, feed_dict={
            self.prev_net_x: inputs
        })
