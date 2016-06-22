# -*- coding: utf-8 -*-
from scipy.spatial.distance import euclidean
from generalizer.state_generalizer import StateGeneralizer


class NormGeneralizer(StateGeneralizer):

    def __init__(self, cellSize):
        super().__init__()
        self.cellSize = cellSize

    def getQState(self, state):
        for s in self.Q:
            if self.isNear(s, state):
                return s
        self.Q[state] = []
        return state

    def isNear(self, o1, o2):
        return euclidean(o1.obs, o2.obs) < self.cellSize