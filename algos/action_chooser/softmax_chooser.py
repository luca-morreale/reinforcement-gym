# -*- coding: utf-8 -*-
from action_chooser.action_chooser import ActionChooser
from numpy import exp


class SoftmaxChooser(ActionChooser):

    def __init__(self, temperature):
        self.temperature = temperature

    # estimate the probability of each action
    def calculateProbabilities(self, actions):
        probs = []
        vals = []
        summation = sum(c.value for c in actions) / self.temperature
        for a in actions:
            probs.append(exp(a.value / self.temperature) / exp(summation))
            vals.append(a.id)
        return probs, vals

