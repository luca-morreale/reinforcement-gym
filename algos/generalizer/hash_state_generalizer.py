# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer


class HashGeneralizer(StateGeneralizer):

    def __init__(self, updater, cellSize):
        super().__init__(updater)
        self.cellSize = cellSize

    def getQState(self, state):
        h = self._hashState(state)
        if h in self.Q:
            return h
        self.Q[h] = []
        return h

    def _hashState(self, s):
        primes = [73856093, 19349663, 83492791, 67867979]
        h = 0
        for i in range(len(s.obs)):
            h += (s.obs[i] / self.cellSize) * primes[i]
        return int(h)
        # hash = (int(pos.x / cellSize) * 73856093) ^ (int(pos.y / cellSize)
        # * 19349663) ^ (int(pos.z / cellSize) * 83492791);
