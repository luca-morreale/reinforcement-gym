# -*- coding: utf-8 -*-
from generalizer.adaptive.adaptive_tile import AdaptiveTile
from generalizer.adaptive.adaptive_tile import Tile
import numpy as np


class Tiling(Tile):

    def __init__(self, num_tiles, n, m, cellSize, coordGenerator):
        self.tiling = []
        self.n = n
        self.m = m
        self.cellSize = cellSize
        self.coordGenerator = coordGenerator
        self._initTiles(num_tiles)

    def _initTiles(self, num_tiles):
        lower = np.zeros(self.n)
        self.tiling = Tile(self.n, self.m, lower, self.cellSize,
                                                        self.coordGenerator)
        '''for i in range(num_tiles):
            self.tiling.append(Tile(self.n, self.m, lower, self.cellSize,
                                                        self.coordGenerator))
            lower = np.copy(lower) + self.cellSize
        '''

    def updateValue(self, state_action, delta):
        self.tiling.updateValue(state_action, delta)

    def getValue(self, coord, action):
        return self.tiling.getValue(coord, action)

    def split(self):
        return NotImplementedError()
