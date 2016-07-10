# -*- coding: utf-8 -*-
'''from state_action import StateAction
import numpy as np
from copy import copy


class Tile():

    def __init__(self, n, m, lower, factor, coordGenerator):
        self.n = n
        self.m = m
        self.lower = lower
        self.factor = factor
        self.coordGenerator = coordGenerator
        self.action_value = {}
        self.sub_tiling = None
        self._initDict()

    def _initDict(self):
        for i in range(self.m):
            self.action_value[i] = 0

    def split(self):
        self.sub_tiling = []
        self.sub_tiling.append(self._recoursiveSplit(1, [self.lower[0]]))
        self.sub_tiling.append(self._recoursiveSplit(1,
                                            [self.lower[0] + self.size / 2]))

    def _recoursiveSplit(self, level, lower):
        low = copy(lower)
        low.append(self.lower[level])
        up = copy(lower)
        up.append(self.lower[level] + self.size / 2)
        if level == self.n - 1:
            return [
                Tile(self.n, self.m, np.array(low), self.factor * 2,
                                                        self.coordGenerator),
                Tile(self.n, self.m, np.array(up), self.factor * 2,
                                                        self.coordGenerator)]
        else:
            tiling = []
            tiling.append(self._recoursiveSplit(level + 1, low))
            tiling.append(self._recoursiveSplit(level + 1, up))
            return tiling

    def updateValue(self, state_action, delta):
        if self.sub_tiling is None:
            self.action_value[state_action.action] += delta
        else:
            tile = self._retreiveTile(state_action)
            tile.updateValue(state_action, delta)

    def getValue(self, coord, action):
        if self.sub_tiling is None:
            return self.action_value[action]
        else:
            tile = self._retreiveTile(coord)
            return tile.getValue(coord, action)

    def _retreiveTile(self, state_action):
        if isinstance(state_action, StateAction):
            obs = state_action.obs
        else:
            obs = state_action
        print(self.coordGenerator.getCoordinates(obs))
        coord = self._identifySubtiling(self.coordGenerator.getCoordinates(obs))
        return self._getSubtile(coord)

    def _identifySubtiling(self, coord):
        new_coord = (np.array(coord) - self.lower) * self.factor
        print(new_coord)
        return new_coord

    def _getSubtile(self, coord):
        print(coord)
        tile = self.sub_tiling
        for i in range(self.n):
            tile = tile[int(coord[i])]
        return tile

    def __str__(self):
        out = "Tile[n:" + str(self.n) + ", m:" + str(self.m) + \
                    ", lower" + str(self.lower) + \
                    ", factor: " + str(self.factor) + \
                    ", action_value:" + str(self.action_value) + \
                    ", sub_tiling:" + str(self.sub_tiling) + "]\t"
        return out
'''

from generalizer.adaptive.quadtree.quadtree import Node


class Tile(Node):

    # None, list of bounds minx,minz,maxx,maxz
    def __init__(self, parent, rect, m):
        super().__init__(parent, rect)
        self.m = m
        self.action_value = {}
        self._initDict()

    def _initDict(self):
        for i in range(self.m):
            self.action_value[i] = 0

    def updateValue(self, state_action, delta):
        if not self.hasChildren():
            self.action_value[state_action.action] += delta
        else:
            for c in self.children:
                # in the future should be fixed!
                if c.contains(state_action.obs[0], state_action.obs[1]):
                    c.updateValue(state_action, delta)

    def getValue(self, coord, action):
        if not self.hasChildren():
            return self.action_value[action]
        else:
            for c in self.children:
                if c.contains(coord[0], coord[1]):
                    return c.getValue(coord, action)

    def hasChildren(self):
        return self.children[0] is not None

    def getinstance(self, rect):
        return Tile(self, rect, self.m)
