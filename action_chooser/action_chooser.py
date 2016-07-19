# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import random_sample


class ActionChooser:

    """ Choose and action among the given following the probabilities
            given by {calculateProbabilities}

    Args:
        actions:    numpy one-dimensional array containing the value of the
                        action.

    Returns:
        index of the action selected.
    """
    def chooseAction(self, actions):
        probs = self.calculateProbabilities(actions)
        return self.weighted_values(probs)

    """ Calculate the probability for each action.

    Args:
        actions:    numpy one-dimensional array

    Returns:
        numpy one-dimensional array containing the respective probability
    """
    def calculateProbabilities(self, actions):
        return NotImplementedError()

    """ Extract a random number and locate the rigth correspondance.
    Args:
        probabilities:    probabilities of the actions

    Returns:
        index of the choosen action.
    """
    def weighted_values(self, probabilities):
        bins = np.add.accumulate(probabilities)
        return np.digitize(random_sample(1)[0], bins)

    def newEpisode(self):
        pass