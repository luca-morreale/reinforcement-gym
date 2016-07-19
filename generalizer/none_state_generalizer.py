# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer
from state_action import StateAction


class NoneGeneralizer(StateGeneralizer):

    def __init__(self, m):
        super().__init__(m)

    def getRepresentation(self, state_action):
        for s in self.Q:
            if s == state_action:
                return s
        self.Q[state_action] = 0
        return state_action

    def getQValue(self, state_action):
        index = self.getRepresentation(state_action)
        return self.Q[index]

    def addDeltaToQValue(self, state_action, value):
        if isinstance(state_action, StateAction):
            index = self.getRepresentation(state_action)
        else:
            return
        self.Q[index] += value

    def prettyPrintQ(self):
        for key, value in enumerate(self.Q):
            print((str(key) + "-> ", value))