# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer
from action import Action
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
    def __init__(self, updater, num_tilings, num_tiles, obs_space, m, n):
        super().__init__(updater)
        self.rndseq = np.random.randint(0, 2 ** 32 - 1, 2048)
        self.tile_vals = np.zeros(n)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.obs_space = obs_space
        self.m = m
        self.n = n

    def getQState(self, state):
        state_vars = []
        for i, var in enumerate(state.obs):
            obs_range = (self.obs_space.high[i] - self.obs_space.low[i])
            if obs_range == float('inf'):
                obs_range = 1
            state_vars.append(var / obs_range * self.num_tiles)

        tiles = self._getTiles(state_vars)
        diff = set(tiles) - set(self.Q.keys())
        for tile in diff:
            self.Q[tile] = []
        return tiles

    def getActionsFor(self, state):
        tiles = self.getQState(state)
        actions = []
        for i in tiles:
            acts = self.Q[i]
            self._accumulateAction(actions, acts)

        return actions

    def _accumulateAction(self, actions, action):
        common_acts = list(set(actions).intersection(action))
        different_acts = list(set(actions) - set(action))

        for act in different_acts:
            actions.append(Action(act.id, act.id))

        for act in common_acts:
            val = next((val for val in actions if val == act), None)
            val.value += act.value

    # translated from https://web.archive.org/web/20030618225322/http://envy.cs.umass.edu/~rich/tiles.html
    def _getTiles(self, variables):
        num_coordinates = len(variables) + 2
        coordinates = [0 for i in range(num_coordinates)]

        qstate = [0 for i in range(len(variables))]
        base = [0 for i in range(len(variables))]
        tiles = [0 for i in range(self.num_tilings)]

        for i, variable in enumerate(variables):
            qstate[i] = int(math.floor(variable * self.num_tilings))
            base[i] = 0

        for j in range(self.num_tilings):
            for i in range(len(variables)):
                if (qstate[i] >= base[i]):
                    coordinates[i] = qstate[i] - ((qstate[i] - base[i])
                                                            % self.num_tilings)
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