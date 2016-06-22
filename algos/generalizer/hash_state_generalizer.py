# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer


class HashGeneralizer(StateGeneralizer):

    def __init__(self, cellSize):
        super().__init__()
        self.cellSize = cellSize

    def getQState(self, Q, state):
        h = self.hashState(state)
        if h in Q:
            #print("collision: ", str(h), " ~ ", str(state))
            return h
        Q[h] = []
        return h

    def getActionOf(self, Q, state, action):
        for a in Q[state]:
            if a == action:
                return a
        Q[state].append(action)
        return action

    def hashState(self, s):
        primes = [73856093, 19349663, 83492791, 67867979]
        h = 0
        for i in range(len(s.obs)):
            h += (s.obs[i] / self.cellSize) * primes[i]
        return int(h)
        # hash = (int(pos.x / cellSize) * 73856093) ^ (int(pos.y / cellSize)
        # * 19349663) ^ (int(pos.z / cellSize) * 83492791);

