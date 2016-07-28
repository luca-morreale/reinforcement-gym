# -*- coding: utf-8 -*-


class Trace:

    def __init__(self, lambda_, gamma):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.e = {}

    ''' Updates the trace of the states occured.
    Args:
        state_action:    last pair state action occured.
    '''
    def updateTrace(self, state_action):
        self._updateTrace()
        if isinstance(state_action, list):
            for s in state_action:
                self.e[s] = 1
        else:
            self.e[state_action] = 1

    def _updateTrace(self):
        for s in self.e:
            self.e[s] *= self.gamma * self.lambda_
            if self.e[s] < 0.09:
                self.e[s] = 0
        self.e = {k: v for k, v in self.e.items() if v > 0}

    def getTrace(self):
        return self.e

    def newEpisode(self):
        self.e = {}
