
from action_chooser import ActionChooser
import numpy as np


# Gaussian Exploration
class GaussianActionChooser(ActionChooser):
    def __init__(self, variance, beta):
        self.variance = variance
        self.beta = beta

    """ Decay the epsilon parameter.
    """
    def newEpisode(self):
        # the reduction of variance is done at each step
        pass

    def reduce_variance(self, delta):
        self.variance = (1 - self.beta) * self.variance + self.beta * delta ** 2

    def _extract_number(self, mean):
        return np.random.normal(mean, self.variance)

    # return a new action similar
    def chooseAction(self, actions):
        action = np.argmax(actions)
        return np.array([self._extract_number(action)])
