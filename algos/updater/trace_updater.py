# -*- coding: utf-8 -*-
from updater.updater import Updater
from updater.trace import Trace


class UpdaterTraced(Updater):

    def __init__(self, discount_factor, learning_rate, lambda_):
        super().__init__(discount_factor, learning_rate)
        self.e = Trace(discount_factor, lambda_)

    def updateStep(self, state, action, vt, rt, estimator):
        self.e.updateTrace(state)
        trace = self.e.getTrace()
        for s in trace:
            delta = estimator.estimateDelta(action.value,
                                [self.alfa, action.visits], self.gamma, vt, rt)
            action.value += trace[s] * delta


