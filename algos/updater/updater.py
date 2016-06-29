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
            self.updateStep(actions[i], rewards[i], estimator, vt[i])

    # update the single value of a pair action-value
    def updateStep(self, state_action, reward, estimator, vt=None):
        state_action.addVisit()
        alfa = [self.alfa, state_action.visits]
        if vt is None:
            state_action.value += estimator.estimateDelta(state_action.value,
                                            alfa, self.gamma, reward)
        else:
            state_action.value += estimator.estimateDelta(state_action.value,
                                        alfa, self.gamma, reward, vt)

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