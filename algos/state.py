# -*- coding: utf-8 -*-
import numpy


class State:

    def __init__(self, obs):
        self.obs = obs
        self.value = 1

    def updateValue(self, updater):
        self.value = updater.estimateStateValue()

    def addVisit(self):
        self.visits += 1

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return numpy.array_equal(self.obs, other.obs)

    def __hash__(self):
        return hash(frozenset(set(self.obs)))

    def __str__(self):
        return "State[obs: " + str(self.obs) + "]"
