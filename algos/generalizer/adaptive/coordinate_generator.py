# -*- coding: utf-8 -*-


class CoordinateGenerator():

    def __init__(self, obs_space, num_tiles):
        self.obs_space = obs_space
        self.num_tiles = num_tiles

    def getCoordinates(self, obs):
        tile_id = []
        for i, var in enumerate(obs):
            obs_range = (self.obs_space.high[i] - self.obs_space.low[i])
            if obs_range == float('inf'):
                obs_range = 1
            # store tile ID
            tile_id.append(var / obs_range * self.num_tiles)
        return tile_id
