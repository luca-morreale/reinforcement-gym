# -*- coding: utf-8 -*-
from action_chooser import ActionChooser
from operator import attrgetter


class EpsilonGreedyChooser(ActionChooser):

    def __init__(self, epsilon, m):
        self.epsilon = epsilon
        self.m = m

    def newEpisode(self):
        self.epsilon = self.epsilon * 0.999  # added epsilon decay

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