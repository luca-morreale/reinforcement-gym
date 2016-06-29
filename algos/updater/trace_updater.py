# -*- coding: utf-8 -*-
from updater.updater import Updater
from updater.trace import Trace


class UpdaterTraced(Updater):

    def __init__(self, discount_factor, learning_rate, lambda_):
        super().__init__(discount_factor, learning_rate)
        self.e = Trace(discount_factor, lambda_)

    def updateStep(self, state_action, reward, estimator, vt=None):
        self.e.updateTrace(state_action)
        trace = self.e.getTrace()
        for sa in trace:
            alfa = [self.alfa, sa.visits]
            delta = estimator.estimateDelta(sa.value, alfa, self.gamma, reward)
            sa.value += trace[sa] * delta

    def newEpisode(self):
        self.e.newEpisode()
        self.pairs = {}