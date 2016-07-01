# -*- coding: utf-8 -*-


class Updater:

    def __init__(self, discount_factor, learning_rate, Q):
        self.gamma = discount_factor
        self.alfa = learning_rate
        self.Q = Q

    # update all values of state-action pair
    def updateEpisode(self, history, estimator):
        states, actions, rewards = history.getSequence()
        vt = self.estimateReturns(rewards)
        for i in range(len(states)):
            self.updateStep(actions[i], rewards[i], estimator, vt[i])

    def update(self, state_action, state_action_value, reward, estimator):
        delta = estimator.estimateDelta(state_action_value, reward, self.gamma)
        self.Q.addDeltaToQValue(state_action, delta * self.alfa)

    # The return is the total discounted reward
    def estimateReturns(self, rewards):
        returns = []
        for i in range(len(rewards)):
            r = 0
            for ii in range(i + 1, len(rewards)):
                r += pow(self.gamma, ii - i) * rewards[i]
            returns.append(r)
        return returns

    def newEpisode(self):
        pass