# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from state_action import StateAction


class StateGeneralizer:

    """ Creates the Generalizer.
    Args:
        updater:    object in charge of update the value of actions
    """
    def __init__(self, m):
        self.Q = {}
        self.m = m

    def getRepresentation(self, state_action):
        return NotImplementedError()

    """ Returns the StateAction estimated value.
    Args:
        state_action:    the state to look for
    Returns:
        number
    """
    def getQValue(self, state_action):
        return NotImplementedError()

    def getCombinedValue(self, state, action):
        return self.getQValue(StateAction(state, action))

    """ Returns an array containing the value of the corrisponding action.
    Args:
        obs:    the state to look for
    Returns:
        array of numbers
    """
    def getPossibleActions(self, obs):
        actions = np.zeros(self.m)
        for i in range(self.m):
            actions[i] = self.getQValue(StateAction(obs, i))
        return actions

    """ Update the value of a state-action pair adding the given value.
    Args:
        state_action:    object representing the state-action
        value:           value to add to the current value
    """
    def addDeltaToQValue(self, state_action, value):
        return NotImplementedError()

    def newEpisode(self):
        pass

    """
        Prints the content of Q in a readable way
    """
    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()
