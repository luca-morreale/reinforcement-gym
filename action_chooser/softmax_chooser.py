# -*- coding: utf-8 -*-
from action_chooser import ActionChooser
import numpy as np


class SoftmaxChooser(ActionChooser):

    def __init__(self, temperature):
        self.temperature = temperature

    # estimate the probability of each action
    def calculateProbabilities(self, actions):
        probs = np.zeros(np.size(actions))
        summation = np.sum(np.exp(actions / self.temperature))
        for i, action in np.ndenumerate(actions):
            probs[i] = np.exp(action / self.temperature) / summation
        return probs
