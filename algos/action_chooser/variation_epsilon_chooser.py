# -*- coding: utf-8 -*-
from action_chooser.action_chooser import ActionChooser
import numpy as np
import random


class VariationEpsilonGreedyChooser(ActionChooser):

    def __init__(self, epsilon, env):
        self.epsilon = epsilon
        self.env = env

    def newEpisode(self):
        self.epsilon = self.epsilon * 0.999  # added epsilon decay

    # choose an action following an epsilon-greedy strategy
    def chooseAction(self, actions):
        next_action = np.argmax(actions)
        if np.random.random() < self.epsilon:
            next_action = self.env.action_space.sample()
            #next_action = np.random.randint(0, self.m - 1)
        return next_action