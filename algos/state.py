# -*- coding: utf-8 -*-


class State:

    def __init__(self, obs, cellSize):
        self.obs = obs
        self.value = 1
        self.cellSize = cellSize

    def updateValue(self, updater):
        self.value = updater.estimateStateValue()

    def addVisit(self):
        self.visits += 1

    def __eq__(self, other):

        return other.obs == self.obs

    # spatial hashing
    def __hash__(self):
        return hash(frozenset(set(self.obs)))

    def __str__(self):
        return "State[obs: " + str(self.obs) + "]"
