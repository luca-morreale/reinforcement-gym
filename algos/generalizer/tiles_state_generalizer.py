# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer
from state_action import StateAction
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
        self._initDict()

    def _initDict(self):
        for i in range(self.n):
            self.Q[i] = StateAction(i)

    """Returns a generalized version of the state-action.

    Args:
        state:     normal version of the state
        action:    action

    Returns:
        list of integer representing the state-action
    """
    def getQState(self, state, action):
        state_vars = self._createStateVar(state)
        tiles = self._getTiles(state_vars, action.id)
        return tiles

    """Returns a generalized version of the state.

    Args:
        state:     normal version of the state

    Returns:
        list of integer representing the state
    """
    def _createStateVar(self, state):
        state_vars = []
        for i, var in enumerate(state.obs):
            obs_range = (self.obs_space.high[i] - self.obs_space.low[i])
            if obs_range == float('inf'):
                obs_range = 1
            state_vars.append(var / obs_range * self.num_tiles)
        return state_vars

    """Returns a generalized version of the state-action, but for all
        possible actions.

    Args:
        state:     normal version of the state

    Returns:
        list of integer representing the state
    """
    def getActionsFor(self, state):
        actions = []
        for i in range(self.m):
            tiles = self.getQState(state, Action(i))
            self._accumulateActions(actions, tiles, i)
        return actions

    def _accumulateActions(self, actions, tiles, index):
        val = 0
        for tile in tiles:
            val += self.Q[tile].value
        actions.insert(index, Action(index, val))

    """Update the value for a pair state-action.

    Args:
        state:         normal version of the state
        action:        action
        reward:        reward
        estimator:     object which is in charge to calculate
                        the delta of the update, must have defined estimateDelta
        vt:            accumulated return

    """
    def update(self, state, action, reward, estimator, vt=None):
        tiles = self.getQState(state, action)
        state_action = self._generateList(tiles)
        self.updater.updateStep(state_action, reward, estimator, vt)

    def _generateList(self, tiles):
        actions = []
        for tile in tiles:
            if self.Q[tile] not in actions:
                actions.append(self.Q[tile])
        return actions

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