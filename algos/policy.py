# -*- coding: utf-8 -*-
from history import History  # lint:ok
from action import Action  # lint:ok
from math import pow
from operator import attrgetter
import numpy as np
from numpy.random import random_sample


class Policy:

    def __init__(self, actionChooser, discount_factor, learning_rate):
        self.actionChooser = actionChooser
        self.gamma = discount_factor
        self.alfa = learning_rate
        self.newEpisode()
        self.Q = {}

    def doEpisode(self, env):
        return NotImplementedError()

    def estimateNewValue(self, value, alfa, vt):
        return NotImplementedError()

    def updateSteps(self, steps):
        return NotImplementedError()

    def updateTrace(self, trace):
        return NotImplementedError()

    # update the value of epsilon based on the number of episodes
    def setEpisodeN(self, n):
        self.epsilon = 1 / n

    # return the state equivalent to the given one, in case no match
    # has been found the state will be added
    def getQState(self, state):
        for s in self.Q:
            if s == state:
                return s
        self.Q[state] = []
        return state

    # return the action equivalent to the one given, in case no match has
    # been found the action will be added
    def getActionOf(self, state, action):
        for a in self.Q[state]:
            if a == action:
                return a
        self.Q[state].append(action)
        return action

    # update all values of state-action pair
    def updateEpisode(self):
        states, actions, rewards = self.history.getSequence()
        vt = self.estimateReturns(rewards)
        for i in range(len(states)):
            self.updateStep(states[i], actions[i], vt[i], i)
        self.newEpisode()

    # update the single value of a pair action-value
    def updateStep(self, state, action, vt, t):
        s = self.getQState(state)
        a = self.getActionOf(s, action)
        a.addVisit()
        a.value = self.estimateNewValue(a.value, [self.alfa, a.visits], vt, t)

    # The return is the total discounted reward
    def estimateReturns(self, rewards):
        returns = []
        for i in range(len(rewards)):
            r = 0
            for ii in range(i, len(rewards)):
                r += pow(self.gamma, ii - i) * rewards[i]
            returns.append(r)
        return returns

    # return an action
    def getAction(self, state):
        for s in self.Q:
            if s == state:
                return self.actionChooser.chooseAction(self.Q[s])
        return Action(self.env.action_space.sample())

    # sets the base values
    def set(self, env, cellSize):
        self.env = env
        self.cellSize = cellSize

    # reset the history
    def newEpisode(self):
        self.history = History()

    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()