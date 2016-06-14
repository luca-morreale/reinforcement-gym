# -*- coding: utf-8 -*-
from action_chooser import ActionChooser
from action import Action
from operator import attrgetter
import numpy as np
from numpy.random import random_sample


class EpsilonGreedyChooser(ActionChooser):

    def __init__(self, epsilon, m):
        self.epsilon = epsilon
        self.m = m

    # choose an action following an epsilon-greedy strategy
    def chooseAction(self, actions):
        prob, values = self.calculateProbabilities(actions)
        val = Action(self.weighted_values(values, prob))
        return val

    # estimate the probability of each action
    def calculateProbabilities(self, actions):
        best_action = max(actions, key=attrgetter('value'))
        probs = []
        vals = []
        for i in range(self.m):
            p = self.epsilon / self.m
            if i == best_action.id:
                p += 1 - self.epsilon
            vals.append(i)
            probs.append(p)
        return probs, vals

    # return the value of the action
    def weighted_values(self, values, probabilities):
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(1)[0], bins)]