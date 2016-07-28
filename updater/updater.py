# -*- coding: utf-8 -*-


class Updater:

    def __init__(self, discount_factor, learning_rate, Q):
        self.gamma = discount_factor
        self.alfa = learning_rate
        self.Q = Q

    ''' Updates value of state-action pair.
    Args:
        state_action:          object representing the pair state action
        state_action_value:    current value of the pair state action to update
        reward:                reward obtained taking the action
        estimator:    object which implements the function ''estimateDelta'
    '''
    def update(self, state_action, state_action_value, reward, estimator):
        delta = estimator.estimateDelta(state_action_value, reward, self.gamma)
        self.Q.addDeltaToQValue(state_action, delta * self.alfa)

    def newEpisode(self):
        pass