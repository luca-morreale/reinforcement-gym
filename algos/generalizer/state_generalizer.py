# -*- coding: utf-8 -*-


class StateGeneralizer:

    # return the state equivalent to the given one, in case no match
    # has been found the state will be added
    def getQState(self, Q, state):
        return NotImplementedError()

    # return the action equivalent to the one given, in case no match has
    # been found the action will be added
    def getActionOf(self, Q, state, action):
        return NotImplementedError()
