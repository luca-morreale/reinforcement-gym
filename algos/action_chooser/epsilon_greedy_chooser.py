# -*- coding: utf-8 -*-
from action_chooser.action_chooser import ActionChooser
import numpy as np


class EpsilonGreedyChooser(ActionChooser):

    def __init__(self, epsilon, m):
        self.epsilon = epsilon
        self.m = m

    """ Decay the epsilon parameter.
    """
    def newEpisode(self):
        self.epsilon = self.epsilon * 0.999

    # estimate the probability of each action
    def calculateProbabilities(self, actions):
        best_action = actions.max()
        probs = np.zeros(np.size(actions))
        for i, action in np.ndenumerate(actions):
            if action == best_action:
                probs[i] = 1 - self.epsilon
            else:
                probs[i] = self.epsilon / (np.size(actions) - 1)
        return probs
