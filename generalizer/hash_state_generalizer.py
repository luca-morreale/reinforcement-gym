# -*- coding: utf-8 -*-
from state_generalizer import StateGeneralizer
from state_action import StateAction
import numpy as np


class HashGeneralizer(StateGeneralizer):

    """Create a generalizer which use spatial hashing.

    Args:
        cellSize:     size of each cell
    """
    def __init__(self, m, cellSize, obs_space):
        StateGeneralizer.__init__(self, m)
        self.cellSize = cellSize
        self.size = len(obs_space.high)
        self.rndseq = np.random.randint(0, 2 ** 32 - 1, self.size + 1)

    def getRepresentation(self, state_action):
        h = self._hashState(state_action)
        if h in self.Q:
            return h
        self.Q[h] = 0
        return h

    def getQValue(self, state_action):
        index = self.getRepresentation(state_action)
        return self.Q[index]

    def _hashState(self, state_action):
        h = 0
        for i in range(self.size):
            h += (state_action.obs[i] / self.cellSize) * self.rndseq[i]
        h += state_action.action * self.rndseq[self.size]
        return int(h)

    """ Update the value of a state-action pair adding the given value.
    Args:
        state_action:    object representing the state-action
        value:           value to add to the current value
    """
    def addDeltaToQValue(self, state_action, value):
        if isinstance(state_action, StateAction):
            index = self.getRepresentation(state_action)
        else:
            index = state_action
        self.Q[index] += value

    def prettyPrintQ(self):
        for key, value in enumerate(self.Q):
            print((str(key) + "-> ", value))