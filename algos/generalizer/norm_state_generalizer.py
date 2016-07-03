# -*- coding: utf-8 -*-
from scipy.spatial.distance import euclidean
from generalizer.state_generalizer import StateGeneralizer
from state_action import StateAction


class NormGeneralizer(StateGeneralizer):

    """Create a generalizer which use spatial hashing.

    Args:
        cellSize:     size of each cell
    """
    def __init__(self, m, cellSize):
        super().__init__(m)
        self.cellSize = cellSize

    def getRepresentation(self, state_action):
        for s in self.Q:
            if self.isNear(s, state_action):
                return s
        self.Q[state_action] = 0
        return state_action

    def getQValue(self, state_action):
        index = self.getRepresentation(state_action)
        return self.Q[index]

    def isNear(self, o1, o2):
        if o1.action == o2.action:
            return euclidean(o1.obs, o2.obs) < self.cellSize
        return False

    """ Update the value of a state-action pair adding the given value.
    Args:
        state_action:    object representing the state-action
        value:           value to add to the current value
    """
    def addDeltaToQValue(self, state_action, value):
        if isinstance(state_action, StateAction):
            index = self.getRepresentation(state_action)
        else:
            return
        self.Q[index] += value

    def prettyPrintQ(self):
        for key, value in enumerate(self.Q):
            print((str(key) + "-> ", value))