# -*- coding: utf-8 -*-
from copy import copy
import numpy as np


class StateAction():

    def __init__(self, obs, action, value=1):
        self.obs = obs
        self.action = copy(action)
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, StateAction):
            return False
        return np.array_equal(other.obs, self.obs) \
                    and other.action == self.action

    def __hash__(self):
        return hash(frozenset(self.obs)) + \
                        31 * hash(self.action)

    def __str__(self):
        return "StateAction[obs: " + str(self.obs) + \
                    "action: " + str(self.action) + \
                    " , value: " + str(self.value) + "]"