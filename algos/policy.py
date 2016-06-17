# -*- coding: utf-8 -*-
from history import History  # lint:ok
from action import Action  # lint:ok


class Policy:

    def __init__(self, actionChooser, generalizer, updater):
        self.actionChooser = actionChooser
        self.generalizer = generalizer
        self.updater = updater
        self.newEpisode()
        self.Q = {}

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
        s = self.generalizer.getQState(self.Q, state)
        if self.Q[s]:
            return self.actionChooser.chooseAction(self.Q[s])
        return Action(self.env.action_space.sample())

    def appendToHistory(self, state, action, reward):
        s = self.generalizer.getQState(self.Q, state)
        self.history.addStep(s, action, reward)

    # sets the base values
    def set(self, env, cellSize):
        self.env = env
        self.cellSize = cellSize

    # reset the history
    def newEpisode(self):
        self.history = History()
        self.actionChooser.newEpisode()

    def prettyPrintQ(self):
        for key in self.Q:
            print(str(key) + "-> ", end="")
            for v in self.Q[key]:
                print(str(v) + " ", end="")
            print()