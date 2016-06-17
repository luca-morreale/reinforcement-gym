# -*- coding: utf-8 -*-
from scipy.spatial.distance import euclidean
from generalizer import StateGeneralizer


class NormGeneralizer(StateGeneralizer):

    def __init__(self, cellSize):
        super().__init__()
        self.cellSize = cellSize

    def getQState(self, Q, state):
        for s in Q:
            if self.isNear(s, state):
                return s
        Q[state] = []
        return state

    def getActionOf(self, Q, state, action):
        for a in Q[state]:
            if a == action:
                return a
        Q[state].append(action)
        return action

    def isNear(self, o1, o2):
        return euclidean(o1.obs, o2.obs) < self.cellSize