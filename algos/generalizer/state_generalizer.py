# -*- coding: utf-8 -*-


class StateGeneralizer:

    def __init__(self):
        self.Q = {}

    """ Returns the state equivalent to the given one, in case no match
        has been found the state will be added.
    Args:
        state:    the state to look for
    Returns:
        an object which identify the state
    """
    def getQState(self, state):
        return NotImplementedError()

    """ Returns the list of actions that can be played in the given state
    Args:
        state:    the state to look for
    Returns:
        list of object of class Action
    """
    def getActionsFor(self, state):
        s = self.getQState(state)
        return self.Q[s]

    # prints the content of Q in a readable way
    """
        Prints the content of Q in a readable way
    """
    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()
