# -*- coding: utf-8 -*-
from action import Action
import numpy as np
from numpy.random import random_sample


class ActionChooser:

    # choose an action following an epsilon-greedy strategy
    def chooseAction(self, actions):
        prob, values = self.calculateProbabilities(actions)
        return Action(self.weighted_values(values, prob))

    def calculateProbabilities(self, actions):
        return NotImplementedError()

    # return the value of the action
    def weighted_values(self, values, probabilities):
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(1)[0], bins)]

    def newEpisode(self):
        pass