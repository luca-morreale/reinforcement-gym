# -*- coding: utf-8 -*-
from history import History  # lint:ok
from action import Action  # lint:ok


class Policy:

    def __init__(self, actionChooser, generalizer, updater):
        self.actionChooser = actionChooser
        self.generalizer = generalizer
        self.updater = updater
        self.newEpisode()

    def doEpisode(self, env):
        return NotImplementedError()

    def estimateDelta(self, value, alfa, gamma, vt, rt):
        return NotImplementedError()

    def updateSteps(self, steps):
        return NotImplementedError()

    def updateTrace(self):
        #self.updater.updateTrace(self.trace, self)
        pass

    # update all values of state-action pair
    def updateEpisode(self):
        self.updater.updateEpisode(self.history, self)
        self.newEpisode()

    # update the single value of a pair action-value
    def updateStep(self, state, action, vt, t):
        self.updater.updateStep(state, action, vt, t, self)

    # return an action
    def getAction(self, state):
        acts = self.generalizer.getActionsFor(state)
        if acts:
            return self.actionChooser.chooseAction(acts)
        return Action(self.env.action_space.sample())

    def appendToHistory(self, state, action, reward):
        s = self.generalizer.getQState(state)
        self.history.addStep(s, action, reward)

    # sets the base values
    def set(self, env, cellSize, show=False):
        self.env = env
        self.cellSize = cellSize
        self.show = show

    # reset the history
    def newEpisode(self):
        self.history = History()
        self.actionChooser.newEpisode()