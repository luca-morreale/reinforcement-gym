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

    def update(self, state, action, reward, estimator, vt=None):
        acts = self.getActionsFor(state)
        state = self.getQState(state)
        diff = list(set([action]) - set(acts))
        self._addNewActions(state, diff)
        self._updateStoredActions(acts + diff, reward, estimator, vt)

    def _updateStoredActions(self, actions, reward, estimator, vt):
        for action in actions:
            self.updater.updateStep(action, reward, estimator, vt)

    def _addNewActions(self, state, action):
        if action:
            self.Q[state].append(action[0])

    def updateEpisode(self, history, estimator):
        self.updater.updateEpisode(history, estimator)

    def newEpisode(self):
        self.updater.newEpisode()

    """
        Prints the content of Q in a readable way
    """
    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()
