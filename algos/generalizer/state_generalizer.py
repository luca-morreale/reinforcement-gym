# -*- coding: utf-8 -*-


class StateGeneralizer:

    """ Creates the Generalizer.
    Args:
        updater:    object in charge of update the value of actions
    """
    def __init__(self, updater):
        self.Q = {}
        self.updater = updater

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
        index = self.getQState(state)
        return self.Q[index]

    def update(self, state, action):
        acts = self.getActionsFor(state)
        for a in acts:
            if a.id == action.id:
                self.updater.update(action)

    def updateEpisode(self, history):
        self.updater.updateEpisode(history)

    """
        Prints the content of Q in a readable way
    """
    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()
