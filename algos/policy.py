# -*- coding: utf-8 -*-
from history import History  # lint:ok
from action import Action  # lint:ok
from math import pow
from operator import attrgetter
import numpy as np
from numpy.random import random_sample


class Policy:

    def __init__(self, discount_factor, learning_rate, epsilon):
        self.gamma = discount_factor
        self.alfa = learning_rate
        self.epsilon = epsilon
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

    #
    def appendActionTo(self, state, action):
        if state in self.Q:
            self.Q[state].append(action)
        else:
            self.Q[state] = [action]

    # update all values of state-action pair
    def updateEpisode(self):
        states, actions, rewards = self.history.getSequence()
        vt = self.estimateReturns(rewards)
        for i in range(len(states)):
            self.updateStep(states[i], actions[i], vt[i], i)
        self.newEpisode()

    # update the single value of a pair action-value
    def updateStep(self, state, action, vt, t, alfa=-1):
        alfa = self.alfa if alfa < 0 else alfa
        if state in self.Q:
            for key in self.Q[state]:
                if action == key:
                    key.addVisit()
                    key.value = self.estimateNewValue(key.value, alfa, vt, t)
                    return
            self.appendActionTo(state, action)
        else:
            action.addVisit()
            action.value = self.estimateNewValue(action.value, alfa, vt, t)
            self.appendActionTo(state, action)

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
                return self.chooseEpsilonGreedy(self.Q[state])
        return Action(self.env.action_space.sample())

    #
    def chooseEpsilonGreedy(self, values):
        prob, actions = self.calculateProbabilities(values)
        a = Action(self.weighted_values(actions, prob))
        print(str(a))
        return a

    #
    def calculateProbabilities(self, values):
        best_action = max(values, key=attrgetter('value'))
        prob = []
        actions = []
        for item in values:
            p = self.epsilon / self.m
            if item == best_action:
                p += 1 - self.epsilon
            actions.append(item.id)
            prob.append(p)
        return prob, actions

    #
    def weighted_values(self, values, probabilities, size=1):
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(size), bins)]

    #
    def set(self, env):
        self.env = env
        self.m = env.action_space.n

    # reset the history
    def newEpisode(self):
        self.history = History()

    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()

    def truncateObservation(self, obs):
        new_obs = []
        for o in obs:
            new_obs.append(float('%.3f' % (o)))
        return new_obs