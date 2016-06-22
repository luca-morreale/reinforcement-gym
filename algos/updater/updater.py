# -*- coding: utf-8 -*-


class Updater:

    def __init__(self, discount_factor, learning_rate):
        self.gamma = discount_factor
        self.alfa = learning_rate

    # update all values of state-action pair
    def updateEpisode(self, history, estimator):
        states, actions, rewards = history.getSequence()
        vt = self.estimateReturns(rewards)
        for i in range(len(states)):
            self.updateStep(states[i], actions[i], vt[i], rewards[i], estimator)

    # update the single value of a pair action-value
    def updateStep(self, state, action, vt, rt, estimator):
        action.addVisit()
        action.value += estimator.estimateDelta(action.value,
                                [self.alfa, action.visits], self.gamma, vt, rt)

    # The return is the total discounted reward
    def estimateReturns(self, rewards):
        returns = []
        for i in range(len(rewards)):
            r = 0
            for ii in range(i, len(rewards)):
                r += pow(self.gamma, ii - i) * rewards[i]
            returns.append(r)
        return returns