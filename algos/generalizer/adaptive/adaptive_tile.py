# -*- coding: utf-8 -*-
from generalizer.adaptive.tile import Tile
import numpy as np


class AdaptiveTile(Tile):

    def __init__(self, parent, rect, m, p):
        super().__init__(parent, rect, m)
        self.u = np.zeros(m)
        self.lowest_update = np.zeros(m) + float('inf')
        self.p = p

    def updateValue(self, state_action, delta):
        super().updateValue(state_action, delta)
        if not self.hasChildren():
            if delta < self.lowest_update[state_action.action]:
                self.u[state_action.action] = 0
                self.lowest_update[state_action.action] = delta
            else:
                self.u[state_action.action] += 1
            if self.u[state_action.action] > self.p:
                self.subdivide()

    def getinstance(self, rect):
        return AdaptiveTile(self, rect, self.m, self.p)
