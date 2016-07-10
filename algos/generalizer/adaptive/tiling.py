# -*- coding: utf-8 -*-
from generalizer.adaptive.adaptive_tile import AdaptiveTile
#from generalizer.adaptive.tile import Tile


class Tiling:

    # obs_space must be a list of bounds
    def __init__(self, m, obs_space, coordGenerator, p):
        self.m = m
        self.p = p
        self.obs_space = obs_space
        self.coordGenerator = coordGenerator
        self.tiling = AdaptiveTile(None, obs_space, self.m, p)

    def updateValue(self, state_action, delta):
        self.tiling.updateValue(state_action, delta)

    def getValue(self, coord, action):
        coord = self.coordGenerator.getCoordinates(coord)
        return self.tiling.getValue(coord, action)

    def split(self):
        return NotImplementedError()
