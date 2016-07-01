# -*- coding: utf-8 -*-
from updater.updater import Updater
from updater.trace import Trace


class UpdaterTraced(Updater):

    def __init__(self, discount_factor, learning_rate, Q, lambda_):
        super().__init__(discount_factor, learning_rate, Q)
        self.trace = Trace(discount_factor, lambda_)

    def update(self, state_action, state_action_value, reward, estimator):
        tiles = self.Q.getRepresentation(state_action)
        self.trace.updateTrace(tiles)
        trace = self.trace.getTrace()
        delta = estimator.estimateDelta(state_action_value, reward, self.gamma)
        for sa in trace:
            self.Q.addDeltaToQValue(sa, delta * self.alfa * trace[sa])

    def newEpisode(self):
        self.e.newEpisode()
        self.pairs = {}