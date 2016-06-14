# -*- coding: utf-8 -*-
from scipy.spatial.distance import euclidean


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
        if not isinstance(other, State):
            return False
        return euclidean(self.obs, other.obs) < self.cellSize

    # spatial hashing
    def __hash__(self):
        return hash(frozenset(set(self.obs)))

    def __str__(self):
        return "State[obs: " + str(self.obs) + "]"
