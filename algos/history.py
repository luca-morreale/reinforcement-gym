# -*- coding: utf-8 -*-


class History:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def addStep(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def getSequence(self):
        return self.states, self.actions, self.rewards
