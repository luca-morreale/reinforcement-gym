# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import random_sample


class ActionChooser:

    # choose an action following an epsilon-greedy strategy
    def chooseAction(self, actions):
        probs = self.calculateProbabilities(actions)
        return self.weighted_values(probs)

    def calculateProbabilities(self, actions):
        return NotImplementedError()

    # return the value of the action
    def weighted_values(self, probabilities):
        bins = np.add.accumulate(probabilities)
        return np.digitize(random_sample(1)[0], bins)

    def newEpisode(self):
        pass