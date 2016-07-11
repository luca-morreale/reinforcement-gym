# -*- coding: utf-8 -*-
from generalizer.state_generalizer import StateGeneralizer
from generalizer.adaptive.tiling import Tiling
from generalizer.adaptive.coordinate_generator import CoordinateGenerator


class AdaptiveTileGeneralizer(StateGeneralizer):

    # obs_space must be a list of bounds
    def __init__(self, m, obs_space, p, indexes):
        super().__init__(m)
        self.coordGenerator = CoordinateGenerator(obs_space, 1)
        obs_space = [obs_space.low[indexes[0]], obs_space.low[indexes[1]],
                    obs_space.high[indexes[0]], obs_space.high[indexes[1]]]
        self.Q = Tiling(m, obs_space, self.coordGenerator, p)
        #self, m, obs_space, coordGenerator, p

    def getQValue(self, state_action):
        return self.Q.getValue(state_action.obs, state_action.action)

    def addDeltaToQValue(self, state_action, value):
        self.Q.updateValue(state_action, value)
