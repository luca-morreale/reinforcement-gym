# -*- coding: utf-8 -*-
from generalizer.adaptive.tile import Tile
import numpy as np


class AdaptiveTile(Tile):

    def __init__(self, n, m, lower, size, coordGenerator):
        super().__init__(n, m, lower, size, coordGenerator)
        self.u = np.array(m)
        self.lowest_update = np.array(m) + float('inf')

    def updateValue(self, state_action, delta):
        super().updateValue(state_action, delta)
        #TODO
