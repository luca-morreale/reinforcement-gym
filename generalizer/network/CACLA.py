
import numpy as np

from regression_network import RegressionNetwork
from action_chooser.gaussian_action_chooser import GaussianActionChooser

# Taken from:
# Reinforcement Learning in Continuous Action Spaces by Hado van Hasselt et al.
class CACLA(object):
    def __init__(self, session, state_dim, learning_rate=0.1, gamma=0.9):
        self.gamma = gamma
        hidden_units = [200, 200, 200, 200, 200]
        self.action_space_explorer = GaussianActionChooser(variance=1.0, beta=0.001)
        self.actor = RegressionNetwork(session, state_dim, hidden_units=hidden_units, learning_rate=learning_rate)
        # critic can also be done with any other generalizer
        self.critic = RegressionNetwork(session, state_dim, hidden_units=hidden_units, learning_rate=learning_rate)

    def get_action(self, state):
        action = self.actor.predict(self.actor.reshape_state(state))
        return self.action_space_explorer.chooseAction(np.array([action]))

    def _calulate_delta(self, state, reward, next_state):
        value_state = self.critic.predict(state)
        value_next_state = self.critic.predict(next_state)
        return reward - self.gamma * value_state + value_next_state

    def train(self, state, action, reward, done, next_state):
        state = self.critic.reshape_state(state)
        next_state = self.critic.reshape_state(next_state)
        delta = self._calulate_delta(state, reward, next_state)
        if delta > 0:
            self.actor.train(state, action)
            self.critic.train(state, self.critic.predict(state) + delta)  # train the value with what??

            self.action_space_explorer.reduce_variance(delta)
