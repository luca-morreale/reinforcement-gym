# -*- coding: utf-8 -*-


class Trace:

    def __init__(self, lambda_, gamma):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.e = {}

    def updateTrace(self, state):
        for s in self.e:
            self.e[s] = self.gamma * self.lambda_ * self.e[s]
            if self.e[s] < 0.01:
                self.e[s] = 0
        self.e[state] = 1

    def getTrace(self):
        return self.e
