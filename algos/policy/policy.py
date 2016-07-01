# -*- coding: utf-8 -*-
from history import History
from state_action import StateAction
import numpy as np


class Policy:

    def __init__(self, actionChooser, Q, updater):
        self.actionChooser = actionChooser
        self.Q = Q
        self.updater = updater
        self.epsilon = 0.1
        self.newEpisode()

    def doEpisode(self, env):
        return NotImplementedError()

    def estimateDelta(self, value, reward, gamma):
        return NotImplementedError()

    """ Returns an integer representing the action to take.

    Args:
        obs:    observation given by the einviroment
    Return:
        number
    """
    def getAction(self, obs):
        # give an array of state-action values
        acts = self.Q.getPossibleActions(obs)
        if acts.any():
            return self.actionChooser.chooseAction(acts)
        return self.env.action_space.sample()
        '''next_action = np.argmax(acts)
        if np.random.random() < self.epsilon:
            next_action = self.env.action_space.sample()
        return next_action'''

    def update(self, obs, action, reward):
        state_action = StateAction(obs, action)
        value = self.Q.getQValue(state_action)
        self.updater.update(state_action, value, reward, self)

    def appendToHistory(self, state, action, reward):
        self.history.addStep(state, action, reward)

    # sets the base values
    def set(self, env, cellSize, show=False):
        self.env = env
        self.cellSize = cellSize
        self.show = show

    # reset the history
    def newEpisode(self):
        self.history = History()
        self.actionChooser.newEpisode()
        self.Q.newEpisode()
        self.epsilon *= 0.999  # added epsilon decay
