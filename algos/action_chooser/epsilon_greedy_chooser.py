# -*- coding: utf-8 -*-
from action_chooser.action_chooser import ActionChooser
import numpy as np


class EpsilonGreedyChooser(ActionChooser):

    def __init__(self, epsilon, m):
        self.epsilon = epsilon
        self.m = m
        self.np_random = np.random.RandomState()

    """ Decay the epsilon parameter.
    """
    def newEpisode(self):
        self.epsilon = self.epsilon * 0.999

    # choose an action following an epsilon-greedy strategy
    def chooseAction(self, actions):
        next_action = np.argmax(actions)
        if np.random.random() < self.epsilon:
            next_action = self.np_random.randint(self.m)
        return next_action