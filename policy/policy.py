# -*- coding: utf-8 -*-
from history import History
from state_action import StateAction


class Policy:

    def __init__(self, actionChooser, Q, updater):
        self.actionChooser = actionChooser
        self.Q = Q
        self.updater = updater
        self.newEpisode()

    """ Defines the procedure to perform an entire episode.

    Args:
        env:    environment on which act.
    """
    def doEpisode(self, env):
        return NotImplementedError()

    """ Calculates the delta for this specific policy.

    Args:
        value:     value of the current state.
        reward:    reward obtained taking the action from this state.
        gamma:     discount factor parameter.
    Returns:
        a number representig the estimated delta.
    """
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

    """ Performs an update just for the current step.

    Args:
        obs:       observation given by the evironment
        action:    action taken, which should be an integer
        reward:    reward obtained playing {action} from state {obs}
    """
    def update(self, obs, action, reward):
        state_action = StateAction(obs, action)
        value = self.Q.getQValue(state_action)
        self.updater.update(state_action, value, reward, self)

    def updateEpisode(self):
        return NotImplementedError()

    """ Appends the tuple to the history for the current episode.

    Args:
        state:     observation given by the evironment
        action:    action taken, which should be an integer
        reward:    reward obtained playing {action} from state {obs}
    """
    def appendToHistory(self, state, action, reward):
        self.history.addStep(state, action, reward)

    """ Sets some useful values.

    Args:
        env:     environment on which play
        show:    boolean, make render or not the environment
    """
    def set(self, env, show=False):
        self.env = env
        self.show = show

    """ Resets the history and recoursively call the
        same function to all children.
    """
    def newEpisode(self):
        self.history = History()
        self.actionChooser.newEpisode()
        self.Q.newEpisode()
