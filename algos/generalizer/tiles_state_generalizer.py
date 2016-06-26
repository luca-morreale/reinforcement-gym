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
        state_vars = self._createStateVar(state)
        tiles = self._getTiles(state_vars)
        diff = set(tiles) - set(self.Q.keys())
        self._addTiles(diff)
        return tiles

    def _createStateVar(self, state):
        state_vars = []
        for i, var in enumerate(state.obs):
            obs_range = (self.obs_space.high[i] - self.obs_space.low[i])
            if obs_range == float('inf'):
                obs_range = 1
            state_vars.append(var / obs_range * self.num_tiles)
        return state_vars

    def _addTiles(self, new_tile):
        for tile in new_tile:
            self.Q[tile] = []

    def getActionsFor(self, state):
        tiles = self.getQState(state)
        actions = []
        for i in tiles:
            self._joinActions(actions, self.Q[i])
        return actions

    def _joinActions(self, actions, q_actions):
        new_actions = list(set(q_actions) - set(actions))
        old_actions = list(set(q_actions) - set(new_actions))

        self._appendActionToTile(actions, new_actions)
        self._accumulateActions(actions, old_actions)

    def _appendActionToTile(self, actions, new_actions):
        for action in new_actions:
            actions.append(Action(action.id, action.value))

    def _accumulateActions(self, actions, old_actions):
        for action in old_actions:
            val = next((val for val in actions if val == action), None)
            val.value += action.value

    def update(self, state, action, reward, estimator, vt=None):
        tiles = self.getQState(state)
        for tile in tiles:
            acts = self.Q[tile]
            diff = list(set([action]) - set(acts))
            self._addNewActions(tile, diff)
            self._updateStoredActions(tile, acts + diff, reward, estimator, vt)

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