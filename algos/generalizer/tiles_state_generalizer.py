# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer
from state_action import StateAction
import numpy as np
import math


class TilesStateGeneralizer(StateGeneralizer):

    """Create a generalizer which use tiles with the given parameters.

    Args:
        updater:     see parent
        n_tiling:
        n_tiles:
        obs_space:    the observation's space of the problem
        m:            number of possible actions
        n:            number of possible states after the discretization
    """
    def __init__(self, num_tilings, num_tiles, obs_space, m, n):
        super().__init__(m)
        self.rndseq = np.random.randint(0, 2 ** 32 - 1, 2048)
        self.tile_vals = np.zeros(n)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.obs_space = obs_space
        self.n = n
        self._initDict()

    def _initDict(self):
        self.Q = np.zeros(self.n)
        #self.Q += 1

    def getRepresentation(self, state_action):
        state_vars = self._createStateVar(state_action)
        return self._getTiles(state_vars, state_action.action)

    def getQValue(self, state_action):
        tiles = self.getRepresentation(state_action)
        return  self._accumulateValue(tiles)

    def _createStateVar(self, state_action):
        state_vars = []
        for i, var in enumerate(state_action.obs):
            obs_range = (self.obs_space.high[i] - self.obs_space.low[i])
            if obs_range == float('inf'):
                obs_range = 1
            state_vars.append(var / obs_range * self.num_tiles)
        return state_vars

    def _accumulateValue(self, tiles):
        val = 0
        for tile in tiles:
            val += self.Q[tile]
        return val

    """ Update the value of a state-action pair adding the given value.
    Args:
        state_action:    object representing the state-action
        value:           value to add to the current value
    """
    def addDeltaToQValue(self, state_action, value):
        if isinstance(state_action, StateAction):
            tiles = self.getRepresentation(state_action)
        else:
            tiles = [state_action]
        for tile in tiles:
            self.Q[tile] += value / self.num_tilings

    # translated from https://web.archive.org/web/20030618225322/
    #    http://envy.cs.umass.edu/~rich/tiles.html
    def _getTiles(self, variables, hash_value):
        num_coordinates = len(variables) + 2
        coordinates = [0 for i in range(num_coordinates)]
        coordinates[-1] = hash_value

        qstate = [0 for i in range(len(variables))]
        base = [0 for i in range(len(variables))]
        tiles = [0 for i in range(self.num_tilings)]

        for i, variable in enumerate(variables):
            qstate[i] = int(math.floor(variable * self.num_tilings))
            base[i] = 0

        for j in range(self.num_tilings):
            for i in range(len(variables)):
                if (qstate[i] >= base[i]):
                    coordinates[i] = qstate[i] - (
                                    (qstate[i] - base[i]) % self.num_tilings)
                else:
                    coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1)
                                        % self.num_tilings) - self.num_tilings

                base[i] += 1 + (2 * i)
            coordinates[len(variables)] = j
            tiles[j] = self._hashCoordinates(coordinates)

        return tiles

    def _hashCoordinates(self, coordinates):
        total = 0
        for i, coordinate in enumerate(coordinates):
            index = coordinate
            index += (449 * i)
            index %= 2048
            while index < 0:
                index += 2048

            total += self.rndseq[index]

        index = total % self.n
        while index < 0:
            index += self.n

        return index

    def prettyPrintQ(self):
        for key, value in np.ndenumerate(self.Q):
            print((str(key) + "-> ", value))
