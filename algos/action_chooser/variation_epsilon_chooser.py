# -*- coding: utf-8 -*-
from action_chooser.epsilon_greedy_chooser import EpsilonGreedyChooser
import numpy as np


class VariationEpsilonGreedyChooser(EpsilonGreedyChooser):

    def __init__(self, epsilon, m):
        super().__init__(epsilon, m)
        self.np_random = np.random.RandomState()

    # choose an action following an epsilon-greedy strategy
    def chooseAction(self, actions):
        next_action = np.argmax(actions)
        if np.random.random() < self.epsilon:
            next_action = self.np_random.randint(self.m)
        return next_action